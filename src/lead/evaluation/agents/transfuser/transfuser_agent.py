import logging

import matplotlib
import numpy as np
import torch

from lead.api.abstract_driving_agent import AbstractDrivingAgent
from lead.common.logging_setup import setup_logging
from lead.evaluation.agents.transfuser.transfuser_control import TransfuserControlMixin
from lead.policy.transfuser.visualization.agent_prediction_visualizer import (
    AgentPredictionVisualizer,
)

matplotlib.use("Agg")  # non-GUI backend for headless servers

setup_logging()
LOG = logging.getLogger(__name__)

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_entry_point():  # dead: disable
    return "TransfuserAgent"


class TransfuserAgent(TransfuserControlMixin, AbstractDrivingAgent):
    """Driving agent wrapping the TransFuser policy.

    The abstract agent assembles the frame and the policy featurizes it; the
    shared :class:`~lead.evaluation.agents.transfuser.transfuser_control.TransfuserControlMixin`
    tracks the model's predicted plans into controls, and this agent adds the
    CARLA-side visualizations.
    """

    def save_step_visualizations(self, sensor_data: dict) -> None:
        """Save the input, demo and debug images and videos of this step.

        Args:
            sensor_data: Sensor data processed by :meth:`tick`.
        """
        evaluation = self.lead_config.evaluation
        if evaluation.save_path is None:
            return
        if self.step % evaluation.produce_frame_frequency != 0:
            return

        # Visualization of prediction for debugging and video recording
        self.features.update(
            {
                "steer": torch.Tensor([self.control.steer]),
                "throttle": torch.Tensor([self.control.throttle]),
                "brake": torch.Tensor([self.control.brake]).bool(),
                "meters_travelled": torch.Tensor([self.meters_travelled]),
            },
        )

        if not hasattr(self, "video_recorder"):
            return

        # The uncompressed cameras of this tick, as the simulator produced them.
        input_image = np.concatenate(
            [
                sensor_data[f"rgb_{camera_idx}"]
                for camera_idx in range(
                    1,
                    self.lead_config.expert.sensor_rig.num_cameras + 1,
                )
            ],
            axis=1,
        )
        self.video_recorder.save_input_image(input_image)
        self.video_recorder.save_input_video_frame(input_image)

        # Get predicted route and waypoints (if available)
        pred_waypoints = (
            self.agent_prediction.prediction.future_waypoints[0]
            if self.agent_prediction.prediction.future_waypoints is not None
            else None
        )
        target_points = {
            "previous": self.features["previous_target_point"][0].cpu().numpy(),
            "current": self.features["target_point"][0].cpu().numpy(),
            "next": self.features["next_target_point"][0].cpu().numpy(),
        }
        self.video_recorder.save_demo_cameras(pred_waypoints, target_points)
        self.video_recorder.save_grid_image_and_video(
            pred_waypoints=pred_waypoints,
            target_points=target_points,
        )

        # Save abstract debug images
        if evaluation.produce_debug_video or evaluation.produce_debug_image:
            image = AgentPredictionVisualizer(
                lead_config=self.lead_config,
                data=self.features,
                prediction=self.agent_prediction,
            ).visualize()
            image = np.array(image).astype(np.uint8)
            self.video_recorder.save_debug_video_frame(image)
            self.video_recorder.save_debug_image(image)


if __name__ == "__main__":
    transfuser_agent = TransfuserAgent()
