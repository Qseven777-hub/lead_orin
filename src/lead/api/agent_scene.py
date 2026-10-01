"""CARLA-free scene pipeline shared by the CARLA agent and the SIL core.

Turns one tick of the leaderboard's sensor payloads into the policy's model
inputs: the sweep history queues, the :class:`SceneData` contract and the
policy's ``build_features``. It imports no ``carla`` and no ``srunner`` so the
same code builds a scene on the Orin.

The only simulator fact it reads is the map name, which the CARLA agent fills
from the world in ``_init`` and the SIL core receives in its session message.
"""

from __future__ import annotations

import typing
from collections import deque

import cv2
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import DynamicStateSE3, EgoStateSE3, Lidar, Timestamp
from py123d.datatypes.metadata.log_metadata import LogMetadata
from py123d.datatypes.sensors.base_camera import Camera
from py123d.datatypes.sensors.radar import Radar, RadarFeature
from py123d.geometry import PoseSE3, Vector3D

from lead.api.abstract_policy import AbstractPolicy
from lead.api.point_cloud_transforms import (
    lidar_sweep_from_carla_ego_frame,
    radar_returns_from_carla_ego_frame,
)
from lead.api.py123d_log_api import (
    CAMERA_ID_BY_LEAD_INDEX,
    CARLA_LINCOLN_MKZ_2020_METADATA,
    RADAR_ID_BY_LEAD_INDEX,
    RADIAL_VELOCITY_FEATURE,
    ordered_target_points,
)
from lead.api.scene_data import SceneData
from lead.common import carla_to_123d, geometry
from lead.common.driving_state import CarlaSensorData
from lead.common.planning import RoutePlanner
from lead.common.sensors import ransac
from lead.config import LeadConfig


class ScenePipelineMixin:
    """The policy-facing half of a driving agent, without CARLA."""

    # Provided by the composing class: BaseAgent / DrivingStateBase for the
    # CARLA agent, the SIL core for the Orin.
    lead_config: LeadConfig
    policy: AbstractPolicy
    step: int
    map_name: str
    compass: float | None
    localized_position: npt.NDArray
    gps_waypoint_planners_dict: dict[float, RoutePlanner]

    def _setup_scene_pipeline(
        self,
        policy: AbstractPolicy,
        lead_config: LeadConfig,
    ) -> None:
        """Build the sweep history, sweep queues and sensor calibration.

        Args:
            policy: The policy whose inputs this pipeline feeds.
            lead_config: Root config tree.
        """
        self.policy = policy
        policy_config = policy.get_policy_config()

        # Sweep history matching the temporal window the policy trains on; a
        # length of N ticks back means N + 1 buffered sweeps.
        self.lidar_sweep_queue: deque = deque(
            maxlen=policy_config.past_lidar_num_iterations + 1,
        )
        self.radar_sweep_queue: deque = deque(
            maxlen=policy_config.past_radar_num_iterations + 1,
        )
        ransac.remove_ground(
            np.random.rand(1000, 3),
            lead_config.expert,
        )  # Pre-compile numba code

        # The rig's calibration, identical to what the logs are written with.
        self._camera_metadatas = carla_to_123d.build_pinhole_camera_metadatas(
            lead_config.expert,
            CARLA_LINCOLN_MKZ_2020_METADATA,
            perturbation_translation=0.0,
            perturbation_rotation=0.0,
        )
        self._lidar_metadata = carla_to_123d.build_lidar_metadata(lead_config.expert)
        self._radar_metadatas, self._merged_radar_metadata = (
            carla_to_123d.build_radar_metadatas(lead_config.expert)
        )

        # Filled by the CARLA agent from the world map or by the SIL core from
        # the session message.
        self.map_name = getattr(self, "map_name", "")

    def tick(self, sensor_data: CarlaSensorData) -> CarlaSensorData:
        """Pre-process the tick's sensors and record the sweep history.

        Args:
            sensor_data: Raw sensor data provided by the leaderboard.

        Returns:
            The pre-processed sensor data.
        """
        sensor_data = super().tick(sensor_data)  # pyright: ignore[reportAttributeAccessIssue]

        # The scene data carries the per-tick sweeps as the logs store them; the
        # policy's featurization owns the merging and ground removal.
        if self.lidar_sweep_queue.maxlen:
            self.lidar_sweep_queue.append(self._lidar_sweep(sensor_data))
        if (
            self.radar_sweep_queue.maxlen
            and self.lead_config.expert.sensor_rig.use_radars
        ):
            self.radar_sweep_queue.append(self._merged_radar_sweep(sensor_data))
        return sensor_data

    def _lidar_sweep(self, sensor_data: CarlaSensorData) -> Lidar:
        """The tick's lidar sweep in the 123D IMU frame, as the modality the logs store."""
        points = self._quantize(sensor_data["lidar_sweep"])
        return Lidar(
            timestamp=self._timestamp(),
            timestamp_end=self._timestamp(),
            metadata=self._lidar_metadata,
            point_cloud_3d=lidar_sweep_from_carla_ego_frame(
                points,
                self.lead_config,
            ).astype(np.float32),
        )

    def _quantize(
        self,
        points: jt.Float[npt.NDArray, "n 3"],
    ) -> jt.Float[npt.NDArray, "n 3"]:
        """Round a sweep to the precision the logs store it at.

        Training sweeps pass through the writer's laspy quantization; applying it here
        keeps the simulator input free of a train-test mismatch.
        """
        storage = self.lead_config.expert.storage
        precision = np.array(
            [
                storage.point_precision_x,
                storage.point_precision_y,
                storage.point_precision_z,
            ],
        )
        quantized = points.copy()
        quantized[:, :3] = np.round(quantized[:, :3] / precision) * precision
        return quantized

    def build_scene_data(self, sensor_data: CarlaSensorData) -> SceneData:
        """Assemble this tick's simulator data into the policy's input contract.

        Produces the same :class:`~lead.api.scene_data.SceneData` training reads from a
        123D log, minus the privileged fields the simulator cannot provide.

        Args:
            sensor_data: Sensor data pre-processed by :meth:`tick`.

        Returns:
            The scene data of this tick.
        """
        target_points = self._target_points()
        return SceneData(
            cameras=self._cameras(sensor_data),
            lidar_sweeps=dict(enumerate(reversed(self.lidar_sweep_queue))),
            radar_sweeps=(
                dict(enumerate(reversed(self.radar_sweep_queue)))
                if self.lead_config.expert.sensor_rig.use_radars
                else None
            ),
            ego_state=self._ego_state(sensor_data),
            log_metadata=LogMetadata(
                dataset=self.lead_config.expert.data_collection.py123d_dataset,
                split="",
                log_name="",
                location=self.map_name.split("/")[-1],
            ),
            past_ego_positions=np.array(
                self.ego_past_positions[::-1],  # pyright: ignore[reportAttributeAccessIssue]
            ),
            past_ego_yaws=np.array(
                self.ego_past_yaws[::-1],  # pyright: ignore[reportAttributeAccessIssue]
            ),
            previous_target_point=target_points["previous_target_point"],
            target_point=target_points["target_point"],
            next_target_point=target_points["next_target_point"],
        )

    def _ego_state(self, sensor_data: CarlaSensorData) -> EgoStateSE3:
        """The ego state the sensors provide: identity pose, speedometer velocity.

        The simulator's true pose is privileged, so the online ego state
        carries only what the featurization reads of it — the forward speed.
        """
        return EgoStateSE3.from_center(
            center_se3=PoseSE3.identity(),
            metadata=CARLA_LINCOLN_MKZ_2020_METADATA,
            timestamp=self._timestamp(),
            dynamic_state_se3=DynamicStateSE3(
                velocity=Vector3D(x=float(sensor_data["speed"]), y=0.0, z=0.0),
                acceleration=Vector3D(x=0.0, y=0.0, z=0.0),
                angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
            ),
        )

    def _cameras(self, sensor_data: CarlaSensorData) -> list[Camera]:
        """The tick's cameras of the active policy, in stitch order, as 123D modalities.

        Each image passes the storage JPEG round-trip so the model sees the same
        compression artifacts as in training.
        """
        quality = self.lead_config.evaluation.inference.jpeg_quality
        timestamp = Timestamp.from_us(
            round(
                self.step * 1e6 / self.lead_config.expert.simulation.carla_fps,
            ),
        )
        lead_indices = {
            camera_id: index for index, camera_id in CAMERA_ID_BY_LEAD_INDEX.items()
        }

        cameras: list[Camera] = []
        for camera_id in self.policy.input_cameras:
            image = sensor_data[f"rgb_{lead_indices[camera_id]}"]
            _, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            metadata = self._camera_metadatas[camera_id]
            cameras.append(
                Camera(
                    metadata=metadata,
                    image=typing.cast(
                        "npt.NDArray[np.uint8]",
                        cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED),
                    ),
                    camera_to_global_se3=metadata.camera_to_imu_se3,
                    timestamp=timestamp,
                ),
            )
        return cameras

    def _timestamp(self) -> Timestamp:
        """The simulation timestamp of the current step."""
        return Timestamp.from_us(
            round(
                self.step * 1e6 / self.lead_config.expert.simulation.carla_fps,
            ),
        )

    def _radars(self, sensor_data: CarlaSensorData) -> list[Radar] | None:
        """The tick's radars in LEAD radar order, as 123D modalities."""
        if not self.lead_config.expert.sensor_rig.use_radars:
            return None

        timestamp = self._timestamp()
        radars: list[Radar] = []
        for radar_idx in range(
            1,
            self.lead_config.expert.sensor_rig.num_radar_sensors + 1,
        ):
            returns = sensor_data[f"radar{radar_idx}"]
            radars.append(
                Radar(
                    timestamp=timestamp,
                    metadata=self._radar_metadatas[RADAR_ID_BY_LEAD_INDEX[radar_idx]],
                    point_cloud_3d=radar_returns_from_carla_ego_frame(
                        returns[:, :3],
                    ).astype(np.float32),
                    point_cloud_features={
                        RADIAL_VELOCITY_FEATURE: returns[:, 3].astype(np.float32),
                    },
                ),
            )
        return radars

    def _merged_radar_sweep(self, sensor_data: CarlaSensorData) -> Radar:
        """The tick's radar returns of all sensors merged into one 123D radar, like the
        merged radar stream of the logs.
        """
        radars = self._radars(sensor_data) or []
        points = [radar.point_cloud_3d for radar in radars]
        velocities = []
        ids = []
        for radar_idx, radar in enumerate(radars, start=1):
            assert radar.point_cloud_features is not None
            velocities.append(radar.point_cloud_features[RADIAL_VELOCITY_FEATURE])
            ids.append(
                np.full(
                    radar.point_cloud_3d.shape[0],
                    int(RADAR_ID_BY_LEAD_INDEX[radar_idx].value),
                    dtype=np.uint8,
                ),
            )
        return Radar(
            timestamp=self._timestamp(),
            metadata=self._merged_radar_metadata,
            point_cloud_3d=(
                np.concatenate(points, axis=0).astype(np.float32)
                if points
                else np.zeros((0, 3), dtype=np.float32)
            ),
            point_cloud_features={
                RadarFeature.IDS.serialize(): (
                    np.concatenate(ids) if ids else np.zeros((0,), dtype=np.uint8)
                ),
                RADIAL_VELOCITY_FEATURE: (
                    np.concatenate(velocities)
                    if velocities
                    else np.zeros((0,), dtype=np.float32)
                ),
            },
        )

    def _target_points(self) -> dict[str, jt.Float[npt.NDArray, " 2"]]:
        """Plan the previous, current and next target point in the ego frame.

        The pop distance adapts when the route bunches target points together,
        and a far-away next point is dropped.
        """
        controller = self.lead_config.evaluation.controller
        points = self._plan_target_points(controller.route_planner_min_distance)

        if controller.sensor_agent_pop_distance_adaptive and self._points_are_dense(
            points,
        ):
            points = self._plan_target_points(4.0)

        # Ignore the next target point if it's too far away.
        if (
            controller.sensor_agent_skip_distant_target_point
            and np.linalg.norm(points["next_target_point"])
            > controller.sensor_agent_skip_distant_target_point_threshold
        ):
            points["next_target_point"] = points["target_point"]
        return points

    def _plan_target_points(
        self,
        pop_distance: float,
    ) -> dict[str, jt.Float[npt.NDArray, " 2"]]:
        """Read the target points of one route planner into the ego frame."""
        planner: RoutePlanner = self.gps_waypoint_planners_dict[pop_distance]
        previous_target_point, current_target_point, next_target_point = (
            ordered_target_points(
                planner.target_points,
                planner.target_point_index,
            )
        )

        compass = self.compass
        assert compass is not None, "tick() must run before build_scene_data()"

        def transform(
            point: jt.Float[npt.NDArray, " 3"],
        ) -> jt.Float[npt.NDArray, " 2"]:
            return geometry.to_local_frame_2d(
                point[:2],
                self.localized_position,
                compass,
            )

        return {
            "previous_target_point": transform(previous_target_point),
            "target_point": transform(current_target_point),
            "next_target_point": transform(next_target_point),
        }

    @staticmethod
    def _points_are_dense(points: dict[str, jt.Float[npt.NDArray, " 2"]]) -> bool:
        """Whether the route's target points bunch up around the ego."""
        previous_target_point = points["previous_target_point"]
        current_target_point = points["target_point"]
        next_target_point = points["next_target_point"]
        close_to_ego = (
            min(
                np.linalg.norm(previous_target_point),
                np.linalg.norm(current_target_point),
            )
            < 10.0
        )
        return bool(
            close_to_ego
            and (
                np.linalg.norm(current_target_point - next_target_point) < 10.0
                or np.linalg.norm(previous_target_point - current_target_point) < 10.0
            ),
        )
