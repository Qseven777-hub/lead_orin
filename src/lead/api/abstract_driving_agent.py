"""Abstract CARLA leaderboard agent wrapping a driving policy for evaluation."""

import abc
import json
import logging
import os
import shutil
import typing

import carla
import torch
import yaml
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.api.abstract_policy import AbstractPolicy
from lead.api.agent_scene import ScenePipelineMixin
from lead.common.base_agent import BaseAgent, CarlaSensorData
from lead.common.driving_state import ControlCommand
from lead.common.sensors.av_sensor_setup import SensorSpec, av_sensor_setup
from lead.config import load_lead_config
from lead.evaluation.inference.policy_runner import PolicyRunner
from lead.evaluation.recorder.infraction_recorder import InfractionRecorder
from lead.evaluation.recorder.video_recorder import VideoRecorder

LOG = logging.getLogger(__name__)


class AbstractDrivingAgent(
    ScenePipelineMixin,
    BaseAgent,
    autonomous_agent.AutonomousAgent,
    abc.ABC,
):
    """CARLA leaderboard protocol adapter around a driving policy.

    Assembles one :class:`~lead.api.scene_data.SceneData` per tick from the simulator's
    sensors and owns everything policy-agnostic: config, sensor rig, infraction and
    video recording, the :meth:`run_step` skeleton. Subclasses supply the policy and
    turn its predictions into a control (:meth:`setup_policy`, :meth:`compute_control`).

    The CARLA-free half (localization, the scene pipeline, featurization) is inherited
    from :class:`~lead.common.base_agent.BaseAgent` and
    :class:`~lead.api.agent_scene.ScenePipelineMixin`, so it can also run on the Orin
    without CARLA.
    """

    # The leaderboard entry point deliberately shadows `BaseAgent.setup`'s
    # internal (lead_config, sensor_agent) signature.
    def setup(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ) -> None:
        self.config_path = path_to_conf_file
        # Bench2Drive appends "+..." to the checkpoint path.
        checkpoint_dir = path_to_conf_file.split("+")[0]

        # Rebuild the config tree saved during training; env/CLI overrides apply on top.
        with open(
            os.path.join(checkpoint_dir, "config.yaml"),
            encoding="utf-8",
        ) as f:
            stored_config = yaml.safe_load(f)
        # The stored tree describes how the model was trained; how it is
        # evaluated is decided by the running code and its overrides.
        stored_config.pop("evaluation", None)
        lead_config = load_lead_config(
            loaded_config=stored_config,
            raise_on_unknown_key=False,
        )

        self.step = -1
        self.initialized = False
        self.device = torch.device("cuda:0")

        # The policy named by ``policy.target``, loaded from the checkpoint.
        self.policy_runner = PolicyRunner(
            lead_config=lead_config,
            model_path=checkpoint_dir,
            device=self.device,
        )
        self.policy: AbstractPolicy = self.policy_runner.policy
        policy_config = self.policy.get_policy_config()

        super().setup(
            lead_config,
            sensor_agent=True,
            past_window_num_iterations=policy_config.past_window_num_iterations,
        )
        self._setup_scene_pipeline(self.policy, lead_config)

        self.setup_policy(checkpoint_dir)

        self.metric_info = {}
        self.meters_travelled = 0.0

        # Infraction tracking
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
        """Build what driving the wrapped policy needs beyond the policy itself, e.g. trackers.

        Called once during :meth:`setup`, after ``self.policy`` is loaded; no-op by default.

        Args:
            checkpoint_dir: Directory holding the trained model checkpoint(s).
        """

    @abc.abstractmethod
    def compute_control(
        self,
        prediction: typing.Any,
        features: dict[str, typing.Any],
    ) -> ControlCommand:
        """Turn the policy's prediction for this tick into a vehicle control.

        Args:
            prediction: Raw prediction of the wrapped policy.
            features: The batched model inputs the prediction was computed on.

        Returns:
            The CARLA-free control; :meth:`run_step` wraps it in a
            ``carla.VehicleControl``.
        """
        raise NotImplementedError

    def save_step_visualizations(self, sensor_data: CarlaSensorData) -> None:
        """Save per-step visualizations of the last control computation; no-op by default.

        Args:
            sensor_data: Sensor data processed by :meth:`tick`.
        """

    def save_input_log(self) -> None:
        """Dump this step's model inputs when input logging is enabled."""
        evaluation = self.lead_config.evaluation
        if evaluation.save_path is None or not evaluation.produce_input_log:
            return
        torch.save(
            {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in self.features.items()
            },
            os.path.join(
                evaluation.input_log_path,
                f"{str(self.step).zfill(5)}.pth",
            ),
        )

    def set_global_plan(
        self,
        global_plan_gps: list[tuple[dict[str, float], RoadOption]],
        privileged_org_dense_route_world_coord: list[
            tuple[carla.Transform, RoadOption]
        ],
    ) -> None:
        """Store the leaderboard's global plan before the scenario starts.

        The dense world-coordinate route is privileged information for offline logging
        and metrics and must not be used by the driving policy.

        Args:
            global_plan_gps: Global route waypoints in GPS space as provided by
                the leaderboard.
            privileged_org_dense_route_world_coord: Dense global route in world coordinates.
        """
        self.privileged_org_dense_route_world_coord = (
            privileged_org_dense_route_world_coord
        )
        LOG.info(
            "Set global plan with %d waypoints.",
            len(self.privileged_org_dense_route_world_coord),
        )
        super().set_global_plan(global_plan_gps, privileged_org_dense_route_world_coord)

    def set_scenario(self, scenario: typing.Any) -> None:
        """Set the scenario reference to track infractions; called by the leaderboard
        after loading the scenario."""
        self.infraction_recorder.set_scenario(scenario)
        LOG.info(
            "[%s] Scenario reference set for infraction tracking",
            type(self).__name__,
        )

    def _init(self) -> None:
        # Get the hero vehicle and the CARLA world
        self._vehicle: carla.Actor = CarlaDataProvider.get_hero_actor()
        self._world: carla.World = self._vehicle.get_world()
        # The scene's map name is the only world fact the scene pipeline reads.
        self.map_name = self._world.get_map().name

        # Set up video recorder
        self.video_recorder = VideoRecorder(
            config_evaluation=self.lead_config.evaluation,
            vehicle=self._vehicle,
            world=self._world,
            step_counter=self.step,
            lead_config=self.lead_config,
        )

        # Passive SafeComfort (LEAD-CES) observer, when that evaluator is active.
        self.ces_monitor = self._init_ces_monitor()

        self.initialized = True

    def _init_ces_monitor(self):
        """Build the passive SafeComfort observer, if selected; else ``None``.

        The import is deferred and guarded by the evaluator switch so that the
        standard/Bench2Drive/Fail2Drive leaderboards, which do not ship the
        ``leaderboard.evaluation`` package, are unaffected.

        Returns:
            The started monitor, or ``None`` when SafeComfort is not active or
            the observer could not be set up.
        """
        if not self.lead_config.evaluation.is_safecomfort:
            return None
        save_path = self.lead_config.evaluation.save_path
        if save_path is None:
            LOG.warning(
                "[%s] SafeComfort active but SAVE_PATH is unset; skipping observer",
                type(self).__name__,
            )
            return None
        try:
            from leaderboard.evaluation.config import CESConfig
            from leaderboard.evaluation.monitor import SafeComfortMonitor

            monitor = SafeComfortMonitor(
                world=self._world,
                ego_actor=self._vehicle,
                route_id=os.environ.get("BENCHMARK_ROUTE_ID", ""),
                output_dir=str(save_path),
                config=CESConfig.from_env(),
            )
            monitor.start()
            LOG.info("[%s] SafeComfort monitor started", type(self).__name__)
            return monitor
        except Exception:
            LOG.warning(
                "[%s] Failed to start SafeComfort monitor; continuing without it",
                type(self).__name__,
                exc_info=True,
            )
            return None

    def sensors(self) -> list[SensorSpec]:
        return av_sensor_setup(
            config=self.lead_config.expert,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )

    def check_infractions(self) -> None:
        """Check and record infractions for the current rollout step."""
        self.infraction_recorder.check_infractions(
            step=self.step,
            meters_travelled=self.meters_travelled,
        )

    @torch.inference_mode()
    def run_step(
        self,
        sensor_data: CarlaSensorData,
        _,
        __=None,
    ) -> carla.VehicleControl:
        """Drive one simulation step: tick, run the policy, record the outcome.

        Args:
            sensor_data: Raw sensor data provided by the leaderboard.

        Returns:
            The vehicle control to apply this step.
        """
        self.step += 1

        if not self.initialized:
            self._init()
            self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            sensor_data = self.tick(sensor_data)
            return self.control

        # Update video recorder step and demo cameras
        if hasattr(self, "video_recorder"):
            self.video_recorder.update_step(self.step)
            self.video_recorder.move_demo_cameras_with_ego()

        # Need to run this every step for GPS filtering
        sensor_data = self.tick(sensor_data)

        # One featurization path with training: the simulator's tick becomes
        # scene data, the policy turns it into its model inputs.
        scene_data = self.build_scene_data(sensor_data)
        self.features = self.policy.features_to_batch(
            self.policy.build_features(scene_data),
            self.device,
        )
        self.save_input_log()
        prediction = self.policy_runner.forward(self.features)
        command = self.compute_control(prediction, self.features)
        self.control = carla.VehicleControl(
            steer=command.steer,
            throttle=command.throttle,
            brake=command.brake,
        )

        self.meters_travelled += (
            sensor_data["speed"].item()
            * self.lead_config.expert.simulation.carla_frame_rate
        )
        sensor_data["meters_travelled"] = self.meters_travelled

        # CARLA will not let the car drive in the initial frames. This help the filter not get confused.
        if self.step < self.lead_config.expert.simulation.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)

        # Check for infractions at this step
        self.check_infractions()

        # Sample the SafeComfort observer at this step (passive; no control effect)
        monitor = getattr(self, "ces_monitor", None)
        if monitor is not None:
            monitor.tick()

        self.save_step_visualizations(sensor_data)

        # Save metric info if in Bench2Drive mode
        if self.lead_config.evaluation.is_bench2drive and hasattr(
            self,
            "get_metric_info",
        ):
            metric = self.get_metric_info()
            self.metric_info[self.step] = metric
            with open(
                f"{self.lead_config.evaluation.save_path}/metric_info.json",
                "w",
            ) as outfile:
                json.dump(self.metric_info, outfile, indent=4)
        return self.control

    def destroy(self, results: object = None) -> None:
        LOG.info(results)

        # Finalise the SafeComfort observer as a fallback when the evaluator
        # did not already do so (idempotent, so the normal path is a no-op).
        monitor = getattr(self, "ces_monitor", None)
        if monitor is not None and not monitor.finalized:
            try:
                progress = 0.0
                status = ""
                route_id = None
                if results is not None:
                    scores = getattr(results, "scores", {}) or {}
                    progress = scores.get("score_route", 0.0) / 100.0
                    status = getattr(results, "status", "")
                    route_id = getattr(results, "route_id", None)
                monitor.finalize(progress=progress, status=status, route_id=route_id)
            except Exception:
                LOG.warning(
                    "[%s] SafeComfort monitor finalisation failed",
                    type(self).__name__,
                    exc_info=True,
                )

        # Clean up video recorder
        if hasattr(self, "video_recorder"):
            self.video_recorder.cleanup()
