"""SIL remote agent: forward the CARLA sensors to the Orin and apply its controls.

Runs on the CARLA host in place of a local policy. It keeps the CARLA side of an
evaluation — sensor rig, infraction/video/metrics, the leaderboard protocol — but
instead of featurizing and running a policy locally, it ships each raw sensor
frame to the Orin over the SIL transport and applies the control that comes back.
The Orin runs the shared
:class:`~lead.evaluation.agents.transfuser.transfuser_core.TransfuserCore`.

Selected with ``LEAD_AGENT_MODULE`` (see ``run_bench2drive_remote_v2.sh``); the
local evaluation scripts are otherwise unchanged.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import typing

import carla
import torch
import yaml
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.api.abstract_driving_agent import AbstractDrivingAgent
from lead.api.py123d_log_api import CAMERA_ID_BY_LEAD_INDEX
from lead.common.base_agent import CarlaSensorData
from lead.common.localization import gps
from lead.common.logging_setup import setup_logging
from lead.config import load_lead_config
from lead.evaluation.recorder.infraction_recorder import InfractionRecorder
from lead.evaluation.sil import codec, contract
from lead.evaluation.sil.transport import SilTransport

setup_logging()
LOG = logging.getLogger(__name__)

DEFAULT_CONTROL_TIMEOUT_MS = 2000
MAX_CONSECUTIVE_TIMEOUTS = 30


def get_entry_point() -> str:
    return "RemoteTransfuserAgent"


class RemoteTransfuserAgent(AbstractDrivingAgent):
    """Forward raw sensor frames to the Orin and apply the returned controls."""

    def setup(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ) -> None:
        self.config_path = path_to_conf_file
        checkpoint_dir = path_to_conf_file.split("+")[0]
        self.checkpoint_dir = checkpoint_dir

        with open(
            os.path.join(checkpoint_dir, "config.yaml"),
            encoding="utf-8",
        ) as handle:
            stored_config = yaml.safe_load(handle)
        stored_config.pop("evaluation", None)
        lead_config = load_lead_config(
            loaded_config=stored_config,
            raise_on_unknown_key=False,
        )

        self.step = -1
        self.initialized = False
        self.lead_config = lead_config
        self.metric_info = {}
        self.meters_travelled = 0.0
        self._consecutive_timeouts = 0
        self._session: dict | None = None

        self.transport = SilTransport(
            os.environ.get("SIL_COMPUTE_OUT", "tcp://127.0.0.1:5560"),
            os.environ.get("SIL_COMPUTE_IN", "tcp://127.0.0.1:5561"),
        )
        self.control_timeout_ms = int(
            os.environ.get("SIL_CONTROL_TIMEOUT_MS", DEFAULT_CONTROL_TIMEOUT_MS),
        )

        lead_indices = {
            camera_id: index for index, camera_id in CAMERA_ID_BY_LEAD_INDEX.items()
        }
        self.camera_indices = tuple(
            lead_indices[camera_id]
            for camera_id in lead_config.policy.transfuser.input_cameras
        )
        self.camera_keys = tuple(
            contract.camera_key(index) for index in self.camera_indices
        )

        self.infraction_recorder = InfractionRecorder(
            config_evaluation=lead_config.evaluation,
            agent_name=type(self).__name__,
        )

        self.track = autonomous_agent.Track.SENSORS

        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg is not installed or not found in PATH. Please install ffmpeg to use video compression.",
            )

    def setup_policy(self, checkpoint_dir: str) -> None:
        """No local policy: the Orin owns the model."""

    def compute_control(
        self,
        prediction: typing.Any,
        features: dict[str, typing.Any],
    ) -> typing.Any:
        """Never called: the Orin computes the control."""
        raise NotImplementedError(
            "the remote agent does not compute controls locally",
        )

    # --- session -------------------------------------------------------------
    def _build_session(self) -> dict:
        """Build the once-per-route session the Orin needs to run the agent."""
        gnss_uses_transverse_mercator = gps.gnss_uses_transverse_mercator(
            CarlaDataProvider.get_client(),
        )
        global_plan = getattr(self, "_global_plan", None) or []
        global_plan_gps = [
            {
                "lat": float(position["lat"]),
                "lon": float(position["lon"]),
                "z": float(position.get("z", 0.0)),
                "command": int(getattr(command, "value", command)),
            }
            for position, command in global_plan
        ]
        lat_ref, lon_ref = gps.find_gps_ref(
            self._global_plan_world_coord,
            global_plan,
        )
        return contract.session(
            route_id=os.environ.get("BENCHMARK_ROUTE_ID", ""),
            scenario_type=os.environ.get("SCENARIO_TYPE", ""),
            map_name=getattr(self, "map_name", ""),
            gnss_uses_transverse_mercator=gnss_uses_transverse_mercator,
            global_plan_gps=global_plan_gps,
            lat_ref=float(lat_ref),
            lon_ref=float(lon_ref),
            camera_indices=self.camera_indices,
            config_source=self.checkpoint_dir,
        )

    # --- remote call ---------------------------------------------------------
    def _wait_control(self, seq: int) -> dict | None:
        """Wait for the control answering ``seq``, dropping stale replies."""
        deadline = time.monotonic() + self.control_timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000.0)
            if remaining_ms <= 0:
                return None
            item = self.transport.recv(remaining_ms)
            if item is None:
                return None
            topic, payload = item
            if topic == contract.TOPIC_CONTROL:
                control = codec.decode(payload)
                if int(control["seq"]) == seq:
                    return control
                LOG.warning(
                    "dropping stale control seq=%s while waiting for %d",
                    control["seq"],
                    seq,
                )
            elif topic == contract.TOPIC_ERROR:
                LOG.error("Orin reported an error: %s", codec.decode(payload))

    def _remote_control(self, sensor_data: CarlaSensorData) -> carla.VehicleControl:
        """Send this tick's sensors and return the control the Orin answers."""
        sensors = {
            key: sensor_data[key] for key in contract.SENSOR_KEYS if key in sensor_data
        }
        for key in self.camera_keys:
            sensors[key] = sensor_data[key]
        self.transport.send(
            contract.TOPIC_SENSOR,
            codec.encode(
                contract.sensor_frame(
                    seq=self.step,
                    step=self.step,
                    sim_time_us=round(
                        self.step * 1e6 / self.lead_config.expert.simulation.carla_fps,
                    ),
                    sensors=sensors,
                    camera_indices=self.camera_indices,
                ),
            ),
        )

        control = self._wait_control(self.step)
        if control is None:
            self._consecutive_timeouts += 1
            LOG.error(
                "no control for step %d (%d consecutive timeouts)",
                self.step,
                self._consecutive_timeouts,
            )
            if self._consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                raise RuntimeError(
                    "the Orin did not answer; aborting so the watchdog can restart",
                )
            # ROS topics are not latched, so the once-per-route session can be
            # published before the Orin's subscription is connected and lost.
            # Re-send it on a timeout so a later step can still succeed.
            if self._session is not None:
                self.transport.send(
                    contract.TOPIC_SESSION,
                    codec.encode(self._session),
                )
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        self._consecutive_timeouts = 0
        return carla.VehicleControl(
            steer=float(control["steer"]),
            throttle=float(control["throttle"]),
            brake=float(control["brake"]),
        )

    # --- leaderboard protocol ------------------------------------------------
    @torch.inference_mode()
    def run_step(
        self,
        sensor_data: CarlaSensorData,
        _,
        __=None,
    ) -> carla.VehicleControl:
        """Forward one tick to the Orin, apply its control and record the outcome."""
        self.step += 1

        if not self.initialized:
            self._init()
            self._session = self._build_session()
            self.transport.send(
                contract.TOPIC_SESSION,
                codec.encode(self._session),
            )
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            return self.control

        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        self.control = self._remote_control(sensor_data)

        self.meters_travelled += (
            float(sensor_data["speed"][1]["speed"])
            * self.lead_config.expert.simulation.carla_frame_rate
        )
        sensor_data["meters_travelled"] = self.meters_travelled

        if self.step < self.lead_config.expert.simulation.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        self.check_infractions()

        monitor = getattr(self, "ces_monitor", None)
        if monitor is not None:
            monitor.tick()

        if self.lead_config.evaluation.is_bench2drive and hasattr(
            self,
            "get_metric_info",
        ):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            import json

            with open(
                f"{self.lead_config.evaluation.save_path}/metric_info.json",
                "w",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)

        return self.control

    def destroy(self, results: object = None) -> None:
        try:
            self.transport.close()
        except Exception:
            LOG.warning("failed to close the SIL transport", exc_info=True)
        super().destroy(results)
