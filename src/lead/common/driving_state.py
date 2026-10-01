"""CARLA-free driving state shared by the expert, the CARLA agent and the SIL core.

Everything here works on plain sensor payloads and configuration: localization
with the Kalman filter, the route planners, the per-tick sensor preprocessing
and the ego history. It deliberately imports no ``carla`` and no ``srunner`` so
the same code can run on the Orin, where CARLA is not installed.

:class:`~lead.common.base_agent.BaseAgent` keeps its CARLA-facing setup (the
GNSS projection flag read from the CARLA client and the ``carla.VehicleControl``
placeholder) and inherits the rest from :class:`DrivingStateBase`.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from lead.common.localization import gps
from lead.common.localization.kalman_filter import KalmanFilter
from lead.common.planning import RoutePlanner
from lead.common.runtime_property_caching import step_cached_property
from lead.common.sensors import point_clouds
from lead.config import LeadConfig

LOG = logging.getLogger(__name__)

# The leaderboard's per-tick sensor map, keyed by sensor id. ``tick`` rewrites it
# in place — raw ``(frame, payload)`` pairs go in, decoded arrays and scalars
# come out — so a key's value type differs before and after that call.
CarlaSensorData = dict[str, Any]


@dataclass
class ControlCommand:
    """A CARLA-free vehicle control, in the ranges CARLA expects."""

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 1.0


class DrivingStateBase:
    """Sensor-derived driving state and its per-tick update, without CARLA."""

    def setup(
        self,
        lead_config: LeadConfig,
        sensor_agent: bool = False,
        past_window_num_iterations: int = 0,
        gnss_uses_transverse_mercator: bool = False,
        global_plan: list[tuple[dict[str, float], Any]] | None = None,
        global_plan_world_coord: list[tuple[Any, Any]] | None = None,
        lat_ref: float | None = None,
        lon_ref: float | None = None,
    ) -> None:
        """Initialize localization, route planners and the history buffers.

        Args:
            lead_config: Root config tree.
            sensor_agent: Whether this is the sensor (driving) agent rather than
                the expert; selects the route planner distances.
            past_window_num_iterations: How many past ticks the policy reads.
            gnss_uses_transverse_mercator: Whether the simulator projects GNSS
                with transverse mercator; on the Orin this comes from the host.
            global_plan: Leaderboard global plan in GPS space.
            global_plan_world_coord: Leaderboard global plan in world coordinates.
            lat_ref: Precomputed GPS latitude reference; computed from the world
                plan when omitted.
            lon_ref: Precomputed GPS longitude reference; computed from the world
                plan when omitted.
        """
        if lat_ref is None or lon_ref is None:
            lat_ref, lon_ref = gps.find_gps_ref(
                global_plan_world_coord,
                global_plan,
            )
        self.noisy_lat_ref, self.noisy_lon_ref = lat_ref, lon_ref
        LOG.info(
            "Noisy lat ref: %s, Noisy lon ref: %s",
            self.noisy_lat_ref,
            self.noisy_lon_ref,
        )
        self.lead_config = lead_config
        self.config_expert = lead_config.expert
        # Cameras ``tick`` preprocesses: the local agent needs all of them for
        # its video, the SIL core only the policy's. Defaults to the whole rig.
        self.camera_indices = tuple(
            range(1, self.config_expert.sensor_rig.num_cameras + 1),
        )
        # Pose history covering the caller's past window — a driving agent
        # passes its policy's, so the sweep alignment always finds the pose of
        # a sweep's own tick age; the expert reads no history and passes none.
        self.kalman_filter = KalmanFilter(
            self.config_expert,
            history_length=past_window_num_iterations + 1,
        )
        self.gnss_uses_transverse_mercator = gnss_uses_transverse_mercator

        self.yaws_queue = deque(
            maxlen=past_window_num_iterations + 1,
        )

        self.control = ControlCommand()
        self.previous_compass: float | None = None
        self.compass: float | None = None

        # --- Route planner ---
        self.sensor_agent = sensor_agent
        route_planner_max_distance = (
            lead_config.evaluation.controller.route_planner_max_distance
            if sensor_agent
            else self.config_expert.simulation.route_planner_max_distance
        )

        self.gps_waypoint_planners_dict: dict[float, RoutePlanner] = {}
        for dist in self.config_expert.simulation.tp_distances:
            planner = RoutePlanner(
                dist,
                route_planner_max_distance,
            )
            planner.set_route(
                global_plan,
                is_gps=True,
                lat_ref=self.noisy_lat_ref,
                lon_ref=self.noisy_lon_ref,
            )
            self.gps_waypoint_planners_dict[dist] = planner

    def tick(self, sensor_data: CarlaSensorData) -> CarlaSensorData:
        # Get the vehicle's speed from sensor
        speed = sensor_data["speed"][1]["speed"]

        # Preprocess the compass data from the IMU
        self.previous_compass = self.compass
        compass = gps.preprocess_compass(
            sensor_data["imu"][1][-1],
        )  # Range [-pi,pi]
        if self.previous_compass is not None:
            compass = float(
                np.unwrap([self.previous_compass, compass])[1],
            )  # Unbounded range
        self.compass = compass
        self.yaws_queue.append(self.compass)

        # Filter the GPS position with Kalman filter
        noisy_gps_pos = gps.convert_gnss_to_carla(
            sensor_data["gps"][1],
            self.noisy_lat_ref,
            self.noisy_lon_ref,
            self.gnss_uses_transverse_mercator,
        )
        self.filtered_state = self.kalman_filter.step(
            noisy_position=noisy_gps_pos,
            compass=self.compass,
            speed=speed,
            control=self.control,
        )
        self.filtered_history = np.array([self.kalman_filter.history_x]).reshape(-1, 4)[
            :,
            :2,
        ]

        # The ego localizes itself, never on the ground truth pose: with the
        # filter, or on the raw noisy GPS the filter takes as its measurement.
        self.localized_position = (
            self.filtered_state[:2]
            if self.config_expert.simulation.use_kalman_filter
            else noisy_gps_pos[:2]
        )
        for planner in self.gps_waypoint_planners_dict.values():
            planner.run_step(np.append(self.localized_position, noisy_gps_pos[2]))

        # Create a dictionary containing the vehicle's state
        sensor_data.update(
            {
                "theta": self.compass,
                "localized_position": self.localized_position,
                "speed": speed,
            },
        )

        # --- LiDAR (the rig always has two) ---
        sensor_data["lidar"] = np.concatenate(
            (
                point_clouds.lidar_to_ego_coordinate(
                    self.config_expert.sensor_rig.lidar_rot_1,
                    self.config_expert.sensor_rig.lidar_pos_1,
                    sensor_data["lidar1"],
                ),
                point_clouds.lidar_to_ego_coordinate(
                    self.config_expert.sensor_rig.lidar_rot_2,
                    self.config_expert.sensor_rig.lidar_pos_2,
                    sensor_data["lidar2"],
                ),
            ),
            axis=0,
        )
        lidar_x, lidar_y = sensor_data["lidar"][:, 0], sensor_data["lidar"][:, 1]
        # Remove lidar points inside ego bounding boxes. We already need LiDAR for expert.
        sensor_data["lidar"] = sensor_data["lidar"][
            (np.abs(lidar_x) > self.config_expert.simulation.ego_extent_x)
            & (np.abs(lidar_y) > self.config_expert.simulation.ego_extent_y)
        ]
        # The raw sweep is what gets stored and what the expert's box labels
        # count hits on; model-specific processing (ground removal, radar
        # merging) is the driving agent's and the data loader's business.
        sensor_data["lidar_sweep"] = sensor_data["lidar"]

        # --- Radar ---
        if self.config_expert.sensor_rig.use_radars:
            for i, radar_calibration in enumerate(
                self.config_expert.sensor_rig.radars,
                start=1,
            ):
                sensor_data[f"radar{i}"] = point_clouds.radar_points_to_ego(
                    sensor_data[f"radar{i}"][1],
                    sensor_pos=radar_calibration["pos"],
                    sensor_rot=radar_calibration["rot"],
                )

        # --- Process camera images ---
        # CARLA cameras
        for camera_idx in self.camera_indices:
            # The expert's RGB cameras only produce data on save ticks
            if f"rgb_{camera_idx}" not in sensor_data:
                assert not self.sensor_agent, f"rgb_{camera_idx} missing"
                continue
            sensor_data[f"rgb_{camera_idx}"] = sensor_data[f"rgb_{camera_idx}"][1][
                :,
                :,
                :3,
            ]
        return sensor_data

    @step_cached_property
    def ego_past_positions(self) -> tuple[tuple[float, float], ...]:
        """Ego past positions in the current ego frame, oldest to current step."""
        past_states = self.filtered_history[:, :2]
        if len(past_states) == 0 or self.compass is None:
            return ()
        current_pos = past_states[-1]
        current_yaw = self.compass
        R_world_to_current = np.array(
            [
                [np.cos(-current_yaw), -np.sin(-current_yaw)],
                [np.sin(-current_yaw), np.cos(-current_yaw)],
            ],
        )
        return tuple(
            (x, y)
            for x, y in ((past_states - current_pos) @ R_world_to_current.T).tolist()
        )

    @step_cached_property
    def ego_past_yaws(self) -> tuple[float, ...]:
        """Ego past yaws in the current ego frame, oldest to current step."""
        if len(self.yaws_queue) == 0:
            return ()
        past_ego_yaws = []
        yaw_now = self.yaws_queue[-1]
        for yaw in self.yaws_queue:
            dyaw = (yaw - yaw_now + np.pi) % (2 * np.pi) - np.pi
            past_ego_yaws.append(dyaw)
        return tuple(past_ego_yaws)
