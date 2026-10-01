"""CARLA-free driving agent core, the compute half of the SIL split.

The CARLA agent (:class:`~lead.api.abstract_driving_agent.AbstractDrivingAgent`)
and the Orin both build the same scene and run the same policy; this class is
that shared orchestration with no simulator dependency. The CARLA agent keeps
its own file and applies the returned control to the vehicle; the Orin runs this
core behind the SIL transport.

Policy-specific behaviour (trackers and turning predictions into controls) is
supplied by a subclass, e.g.
:class:`~lead.evaluation.agents.transfuser.transfuser_core.TransfuserCore`.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Callable

import torch

from lead.api.agent_scene import ScenePipelineMixin
from lead.common.driving_state import (
    CarlaSensorData,
    ControlCommand,
    DrivingStateBase,
)
from lead.config import LeadConfig

LOG = logging.getLogger(__name__)


class PolicyAgentCore(ScenePipelineMixin, DrivingStateBase):
    """Run one policy in closed loop without CARLA."""

    def setup_core(
        self,
        *,
        lead_config: LeadConfig,
        policy: typing.Any,
        forward: Callable[[typing.Any], typing.Any],
        global_plan: list[tuple[dict[str, float], typing.Any]] | None,
        global_plan_world_coord: list[tuple[typing.Any, typing.Any]] | None,
        gnss_uses_transverse_mercator: bool,
        map_name: str,
        device: torch.device,
        lat_ref: float | None = None,
        lon_ref: float | None = None,
        camera_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize the state, the scene pipeline and the policy.

        Args:
            lead_config: Root config tree, matching the host's.
            policy: The policy whose features and forward this core runs.
            forward: Callable running the policy on a batch (``PolicyRunner.forward``
                on the Orin, or the local runner).
            global_plan: Leaderboard global plan in GPS space.
            global_plan_world_coord: Leaderboard global plan in world coordinates.
            gnss_uses_transverse_mercator: GNSS projection flag from the host.
            map_name: CARLA map name, replacing ``self._world.get_map().name``.
            device: Device to place the batched inputs on.
            lat_ref: Precomputed GPS latitude reference, when the host sent one.
            lon_ref: Precomputed GPS longitude reference, when the host sent one.
            camera_indices: Leaderboard camera indices the core preprocesses;
                defaults to the whole rig when omitted.
        """
        self.step = -1
        self.device = device
        self.forward = forward
        self.map_name = map_name
        policy_config = policy.get_policy_config()

        DrivingStateBase.setup(
            self,
            lead_config=lead_config,
            sensor_agent=True,
            past_window_num_iterations=policy_config.past_window_num_iterations,
            gnss_uses_transverse_mercator=gnss_uses_transverse_mercator,
            global_plan=global_plan,
            global_plan_world_coord=global_plan_world_coord,
            lat_ref=lat_ref,
            lon_ref=lon_ref,
        )
        if camera_indices is not None:
            # Only the cameras the host forwarded; the core has no other images.
            self.camera_indices = tuple(camera_indices)
        self._setup_scene_pipeline(policy, lead_config)
        self.setup_policy("")
        self.meters_travelled = 0.0

    def setup_policy(self, checkpoint_dir: str) -> None:
        """Build the policy-specific trackers; a subclass overrides this."""
        raise NotImplementedError

    def compute_control(
        self,
        prediction: typing.Any,
        features: dict[str, typing.Any],
    ) -> ControlCommand:
        """Turn a prediction into a control; a subclass overrides this."""
        raise NotImplementedError

    @torch.inference_mode()
    def run_step(self, sensor_data: CarlaSensorData) -> ControlCommand:
        """Run one closed-loop step from raw sensor payloads to a control.

        Args:
            sensor_data: Raw per-sensor payloads, exactly as the leaderboard
                hands them to ``run_step``.

        Returns:
            The CARLA-free control for this step.
        """
        self.step += 1
        sensor_data = self.tick(sensor_data)
        scene_data = self.build_scene_data(sensor_data)
        self.features = self.policy.features_to_batch(
            self.policy.build_features(scene_data),
            self.device,
        )
        prediction = self.forward(self.features)
        command = self.compute_control(prediction, self.features)
        self.meters_travelled += (
            float(sensor_data["speed"])
            * self.lead_config.expert.simulation.carla_frame_rate
        )
        return command
