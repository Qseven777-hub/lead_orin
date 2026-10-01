"""Camera input geometry of the TransFuser model."""

from py123d.datatypes.sensors.base_camera import CameraID

from lead.config.node import ConfigNode

# The two camera sets the policy can ingest, in left-to-right stitch order. The
# surround set follows the sensor rig's index order (front-left, front,
# front-right, rear-right, rear, rear-left).
_FRONT_CAMERA_IDS: tuple[CameraID, ...] = (
    CameraID.PCAM_L0,
    CameraID.PCAM_F0,
    CameraID.PCAM_R0,
)
_SURROUND_CAMERA_IDS: tuple[CameraID, ...] = (
    CameraID.PCAM_L0,
    CameraID.PCAM_F0,
    CameraID.PCAM_R0,
    CameraID.PCAM_R1,
    CameraID.PCAM_B0,
    CameraID.PCAM_L1,
)


class TransfuserCameraConfig(ConfigNode):
    """Camera selection and the stitched image geometry."""

    # Whether the policy ingests the full six-camera surround rig instead of
    # the three front cameras. A bool rather than a camera list so it can be
    # overridden from env/CLI -- a list of CameraID cannot, because string
    # overrides do not convert back to the enum.
    use_six_cameras: bool = False

    @property
    def input_cameras(self) -> list[CameraID]:
        """Cameras the model ingests, in left-to-right stitch order."""
        ids = _SURROUND_CAMERA_IDS if self.use_six_cameras else _FRONT_CAMERA_IDS
        return list(ids)

    @property
    def final_image_width(self) -> int:
        """Final width of the stitched model input across the input cameras."""
        return len(self.input_cameras) * self._root.expert.sensor_rig.camera_width

    @property
    def final_image_height(self) -> int:
        """Final height of images."""
        return self._root.expert.sensor_rig.image_height

    @property
    def img_vert_anchors(self) -> int:
        """Number of vertical anchors for image feature maps."""
        return self.final_image_height // 32

    @property
    def img_horz_anchors(self) -> int:
        """Number of horizontal anchors for image feature maps."""
        return self.final_image_width // 32
