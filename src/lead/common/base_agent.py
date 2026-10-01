"""Base agent: the CARLA-facing pieces of the driving state.

The CARLA-free logic (localization, route planners, per-tick preprocessing, ego
history) lives in :class:`~lead.common.driving_state.DrivingStateBase`. This
class only adds what needs the simulator: the GNSS projection flag read from the
CARLA client and the ``carla.VehicleControl`` placeholder the Kalman filter reads.
"""

import logging
from typing import Any

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common.driving_state import CarlaSensorData, DrivingStateBase
from lead.common.localization import gps
from lead.config import LeadConfig

LOG = logging.getLogger(__name__)

__all__ = ["BaseAgent", "CarlaSensorData"]


class BaseAgent(DrivingStateBase):
    """Agent class handle basic sensor processing that both expert and student need."""

    # Set by the composed `AutonomousAgent` base (CARLA leaderboard) via
    # `set_global_plan`, not by this class itself.
    _global_plan_world_coord: list[tuple[carla.Transform, Any]]
    _global_plan: list[tuple[dict[str, float], Any]]

    def setup(
        self,
        lead_config: LeadConfig,
        sensor_agent: bool = False,
        past_window_num_iterations: int = 0,
    ) -> None:
        """Initialize the CARLA-free state, taking the GNSS flag from CARLA.

        Args:
            lead_config: Root config tree.
            sensor_agent: Whether this is the driving agent, not the expert.
            past_window_num_iterations: How many past ticks the policy reads.
        """
        gnss_uses_transverse_mercator = gps.gnss_uses_transverse_mercator(
            CarlaDataProvider.get_client(),
        )
        super().setup(
            lead_config,
            sensor_agent,
            past_window_num_iterations,
            gnss_uses_transverse_mercator,
            self._global_plan,
            self._global_plan_world_coord,
        )
        self.control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
