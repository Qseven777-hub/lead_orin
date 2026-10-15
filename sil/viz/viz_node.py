#!/usr/bin/env python3
"""Republish the SIL topics as the standard message types RViz and rqt_plot read.

The SIL run carries everything as ``std_msgs/UInt8MultiArray`` holding msgpack
bytes, which RViz cannot display. This node subscribes to those topics (live or
played back from a recorded bag) and republishes typed messages under one
prefix: LiDAR and radar ``PointCloud2``, plan and route ``nav_msgs/Path``,
target points / ego box / control readout ``MarkerArray``, camera ``Image``,
the control vector for ``rqt_plot``, and a ``map -> base_link`` transform
recovered from the GNSS.

It runs beside ``roscore`` in the ROS Noetic environment, so the Python 3.10
compute process never has to import rospy. The same displays carry over to a
real vehicle, where the drivers publish the same typed topics.

Environment (all optional)::

    SIL_VIZ_SENSOR   sensor-frame topic    (default lead/sensor_frame)
    SIL_VIZ_PLAN     plan topic            (default lead/plan)
    SIL_VIZ_CONTROL  control topic         (default lead/control)
    SIL_VIZ_SESSION  session topic         (default lead/session)
    SIL_VIZ_PREFIX   output topic prefix   (default lead/viz)
    SIL_VIZ_ODOM     publish map->base_link (default 1)
"""

from __future__ import annotations

import importlib.util
import math
import os
import typing
from collections import deque
from pathlib import Path as FsPath

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from visualization_msgs.msg import Marker, MarkerArray

try:  # a stock ROS/RViz install always ships tf2_ros; keep it a soft dependency
    import tf2_ros
except ImportError:  # pragma: no cover - minimal RoboStack envs
    tf2_ros = None  # type: ignore[assignment]

# Sensor extrinsics of the SIL checkpoint (``expert.sensor_rig``). The bag holds
# each sweep in its own sensor frame, so the ego-frame publish applies the same
# rotation/translation the policy applies.
LIDAR_POS = ((1.0, 0.0, 2.5), (-1.0, 0.0, 2.5))
LIDAR_ROT = ((0.0, 0.0, -90.0), (0.0, 0.0, -270.0))
RADAR_POS = (
    (2.6, 0.0, 0.6),
    (2.6, 0.0, 0.6),
    (-2.6, 0.0, 0.6),
    (-2.6, 0.0, 0.6),
)
RADAR_ROT = (
    (0.0, 0.0, -45.0),
    (0.0, 0.0, 45.0),
    (0.0, 0.0, 135.0),
    (0.0, 0.0, 225.0),
)
EGO_EXTENT_X = 2.45
EGO_EXTENT_Y = 1.06

# Global plan RoadOption values, for the navigation-point labels.
ROAD_OPTIONS = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGELANELEFT",
    6: "CHANGELANERIGHT",
}

BASE_FRAME = "base_link"
MAP_FRAME = "map"

# Number of GNSS fixes averaged into the published ego pose. The raw fixes
# jitter by about a metre, so a long window is needed both to steady the
# position and to recover the heading (this route averages well under 1 m/s,
# so a short window is pure noise).
POSE_WINDOW = 45

# Constant from CARLA's GNSS simulation, used to invert its transverse
# Mercator projection into map metres.
EARTH_RADIUS_EQUA = 6378137.0
TMERC_K0 = 0.9996


def load_codec() -> typing.Any:
    """Load the SIL wire codec without importing the ``lead`` package.

    ``lead/__init__`` installs a jaxtyping import hook and reads the simulator
    paths, none of which belong in a ROS node, while ``codec`` itself only needs
    msgpack and numpy.

    Returns:
        The loaded ``lead.evaluation.sil.codec`` module.
    """
    repo_root = FsPath(__file__).resolve().parents[2]
    path = repo_root / "src" / "lead" / "evaluation" / "sil" / "codec.py"
    spec = importlib.util.spec_from_file_location("lead_sil_codec", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the SIL codec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_of(value: typing.Any) -> np.ndarray | None:
    """The ndarray inside a raw ``(frame_id, array)`` sensor payload.

    Args:
        value: A sensor entry from the frame's ``sensors`` dict.

    Returns:
        The array, or None when the entry carries none.
    """
    if isinstance(value, (tuple, list)):
        arrays = [item for item in value if isinstance(item, np.ndarray)]
        return arrays[-1] if arrays else None
    return value if isinstance(value, np.ndarray) else None


def to_ego(
    points: np.ndarray,
    pos: tuple[float, float, float],
    yaw_deg: float,
) -> np.ndarray:
    """Rotate sensor-frame points into the ego frame.

    The rig only yaws its sensors, so this is the yaw-only rotation plus the
    sensor translation, mirroring the policy's ``lidar_to_ego_coordinate``.
    Columns beyond ``xyz`` (intensity, Doppler, ...) are carried through.

    Args:
        points: Sensor-frame ``(N, >=3)`` points; the first three columns are
            ``x, y, z`` in metres.
        pos: Sensor position ``(x, y, z)`` in the ego frame.
        yaw_deg: Sensor yaw in degrees.

    Returns:
        The points in the ego frame, with the same number of columns.
    """
    rad = math.radians(yaw_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    x = cos * points[:, 0] - sin * points[:, 1]
    y = sin * points[:, 0] + cos * points[:, 1]
    ego = np.stack([x, y, points[:, 2]], axis=1) + np.asarray(pos, dtype=np.float32)
    if points.shape[1] > 3:
        return np.concatenate([ego, points[:, 3:]], axis=1)
    return ego


def tmerc_to_map(gnss: np.ndarray, lat_ref: float, lon_ref: float) -> np.ndarray:
    """Invert CARLA's transverse-Mercator GNSS reading into map metres.

    Args:
        gnss: ``(lat, lon, alt)`` reading in degrees and metres.
        lat_ref: Latitude reference of the map.
        lon_ref: Longitude reference of the map.

    Returns:
        The local ``(x, y)`` position in metres.
    """
    lat, lon = float(gnss[0]), float(gnss[1])
    phi = math.radians(lat)
    delta_lambda = math.radians(lon - lon_ref)
    b = math.cos(phi) * math.sin(delta_lambda)
    x = 0.5 * TMERC_K0 * EARTH_RADIUS_EQUA * math.log((1 + b) / (1 - b))
    y = (
        TMERC_K0
        * EARTH_RADIUS_EQUA
        * (math.atan(math.tan(phi) / math.cos(delta_lambda)) - math.radians(lat_ref))
    )
    return np.array([x, y])


def mercator_to_map(gnss: np.ndarray, lat_ref: float, lon_ref: float) -> np.ndarray:
    """Invert the leaderboard's equatorial-Mercator GPS into map metres.

    The global plan (from the routes XML) is projected with the equatorial
    Mercator, which is *not* the transverse Mercator CARLA 0.9.16's GNSS sensor
    reports. Using the sensor's projection for the plan puts the route about
    10 km away from the ego.

    Args:
        gnss: ``(lat, lon, alt)`` reading in degrees and metres.
        lat_ref: Latitude reference of the map.
        lon_ref: Longitude reference of the map.

    Returns:
        The local ``(x, y)`` position in metres.
    """
    lat, lon = float(gnss[0]), float(gnss[1])
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (
        EARTH_RADIUS_EQUA * scale
    )
    mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
    y = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
        - my
    )
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    return np.array([x, y])


def cloud2(
    points: np.ndarray,
    names: typing.Sequence[str],
    frame_id: str,
    stamp: rospy.Time,
) -> PointCloud2:
    """Build a ``PointCloud2`` from an ``(N, len(names))`` float32 array.

    Args:
        points: The point data, one column per field.
        names: Field names, in column order.
        frame_id: The TF frame the points are expressed in.
        stamp: Message timestamp.

    Returns:
        The filled message.

    Raises:
        ValueError: If the point columns do not match ``names``; a mismatch
            would make ``point_step`` disagree with the data and every consumer
            would silently drop the message.
    """
    if points.ndim != 2 or points.shape[1] != len(names):
        raise ValueError(
            f"expected {len(names)} point columns {names}, got shape {points.shape}",
        )
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = int(points.shape[0])
    cloud.fields = [
        PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(names)
    ]
    cloud.is_bigendian = False
    cloud.point_step = 4 * len(names)
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = bool(points.shape[0]) and bool(np.isfinite(points).all())
    cloud.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    return cloud


def path2(points: np.ndarray, frame_id: str, stamp: rospy.Time) -> Path:
    """Build a ``nav_msgs/Path`` from an ``(N, 2)`` ego-frame matrix.

    Args:
        points: ``(x, y)`` waypoints in metres.
        frame_id: The TF frame the points are expressed in.
        stamp: Message timestamp.

    Returns:
        The filled message.
    """
    message = Path()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    for x, y in np.asarray(points).reshape(-1, 2):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def image2(image: np.ndarray, frame_id: str, stamp: rospy.Time) -> Image:
    """Build a ``sensor_msgs/Image`` in ``bgr8`` from a BGRA/BGR array.

    Args:
        image: ``(H, W, 3|4)`` uint8 camera frame.
        frame_id: The camera's TF frame.
        stamp: Message timestamp.

    Returns:
        The filled message.
    """
    bgr = np.ascontiguousarray(image[:, :, :3])
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height, message.width = bgr.shape[:2]
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = bgr.tobytes()
    return message


class SilViz:
    """Decode the SIL topics and republish typed messages for RViz."""

    def __init__(self) -> None:
        """Resolve topics, build publishers, and subscribe to the SIL topics."""
        self.codec = load_codec()
        prefix = os.environ.get("SIL_VIZ_PREFIX", "lead/viz").strip("/")
        self._prefix = prefix
        self._publish_odom = os.environ.get("SIL_VIZ_ODOM", "1") not in ("0", "false")

        self._lat_ref = 0.0
        self._lon_ref = 0.0
        self._gnss = deque(maxlen=POSE_WINDOW)
        self._control_text = ""
        self._camera_indices: tuple[int, ...] = ()

        self.pub_lidar = rospy.Publisher(
            f"{prefix}/lidar",
            PointCloud2,
            queue_size=5,
        )
        self.pub_radar = rospy.Publisher(
            f"{prefix}/radar",
            PointCloud2,
            queue_size=5,
        )
        self.pub_path = rospy.Publisher(f"{prefix}/path", Path, queue_size=5)
        self.pub_route = rospy.Publisher(f"{prefix}/route", Path, queue_size=5)
        self.pub_markers = rospy.Publisher(
            f"{prefix}/markers",
            MarkerArray,
            queue_size=5,
        )
        self.pub_control = rospy.Publisher(
            f"{prefix}/control",
            Float32MultiArray,
            queue_size=5,
        )
        self.pub_odom = rospy.Publisher(f"{prefix}/odom", Odometry, queue_size=5)
        self.pub_camera = rospy.Publisher(f"{prefix}/camera/image", Image, queue_size=2)
        # The global navigation plan is static per route, so latch it.
        self.pub_navigation = rospy.Publisher(
            f"{prefix}/navigation",
            Path,
            queue_size=1,
            latch=True,
        )
        self.pub_navigation_points = rospy.Publisher(
            f"{prefix}/navigation_points",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self._camera_publishers: dict[int, rospy.Publisher] = {}
        self._tf = tf2_ros.TransformBroadcaster() if tf2_ros else None
        if self._tf is None:
            rospy.logwarn(
                "tf2_ros missing: not publishing map->base_link; install "
                "ros-noetic-tf2-ros (RViz ships it) or set the fixed frame to "
                "base_link.",
            )

        rospy.Subscriber(
            os.environ.get("SIL_VIZ_SESSION", "lead/session"),
            UInt8MultiArray,
            self._on_session,
            queue_size=5,
        )
        rospy.Subscriber(
            os.environ.get("SIL_VIZ_SENSOR", "lead/sensor_frame"),
            UInt8MultiArray,
            self._on_sensor,
            queue_size=5,
        )
        rospy.Subscriber(
            os.environ.get("SIL_VIZ_PLAN", "lead/plan"),
            UInt8MultiArray,
            self._on_plan,
            queue_size=5,
        )
        rospy.Subscriber(
            os.environ.get("SIL_VIZ_CONTROL", "lead/control"),
            UInt8MultiArray,
            self._on_control,
            queue_size=5,
        )
        rospy.loginfo("lead_sil_viz publishing under %s", prefix)

    def _camera_publisher(self, index: int) -> rospy.Publisher:
        publisher = self._camera_publishers.get(index)
        if publisher is None:
            publisher = rospy.Publisher(
                f"{self._prefix}/camera_{index}/image",
                Image,
                queue_size=2,
            )
            self._camera_publishers[index] = publisher
        return publisher

    def _on_session(self, message: UInt8MultiArray) -> None:
        session = self.codec.decode(bytes(message.data))
        self._lat_ref = float(session.get("lat_ref") or 0.0)
        self._lon_ref = float(session.get("lon_ref") or 0.0)
        self._camera_indices = tuple(session.get("camera_indices") or ())
        rospy.loginfo(
            "session route=%s map=%s cameras=%s",
            session.get("route_id"),
            session.get("map_name"),
            list(self._camera_indices),
        )
        self._publish_navigation(session)

    def _publish_navigation(self, session: dict) -> None:
        """Publish the leaderboard global plan as a map-frame path and points.

        Args:
            session: The decoded session message carrying ``global_plan_gps``.
        """
        plan = session.get("global_plan_gps") or []
        if not plan:
            return
        lat_ref = float(session.get("lat_ref") or 0.0)
        lon_ref = float(session.get("lon_ref") or 0.0)
        points = np.array(
            [
                mercator_to_map(
                    np.array([float(entry["lat"]), float(entry["lon"]), 0.0]),
                    lat_ref,
                    lon_ref,
                )
                for entry in plan
            ],
            dtype=np.float64,
        )
        stamp = rospy.Time.now()
        self.pub_navigation.publish(path2(points, MAP_FRAME, stamp))
        self.pub_navigation_points.publish(
            self._navigation_markers(plan, points, stamp),
        )
        rospy.loginfo("navigation plan: %d points", len(plan))

    def _navigation_markers(
        self,
        plan: list,
        points: np.ndarray,
        stamp: rospy.Time,
    ) -> MarkerArray:
        markers = MarkerArray()
        for index, point in enumerate(points):
            entry = plan[index]
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = MAP_FRAME
            sphere.ns = "navigation_point"
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(point[0])
            sphere.pose.position.y = float(point[1])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.5
            sphere.color.r, sphere.color.g, sphere.color.b = (1.0, 0.0, 0.0)
            sphere.color.a = 1.0
            markers.markers.append(sphere)

            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = MAP_FRAME
            text.ns = "navigation_command"
            text.id = len(plan) + index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(point[0])
            text.pose.position.y = float(point[1])
            text.pose.position.z = 2.5
            text.pose.orientation.w = 1.0
            text.scale.z = 1.2
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            command = entry.get("command")
            text.text = f"{index}: {ROAD_OPTIONS.get(command, command)}"
            markers.markers.append(text)
        return markers

    def _on_control(self, message: UInt8MultiArray) -> None:
        control = self.codec.decode(bytes(message.data))
        infer_ms = control.get("infer_ms") or 0.0
        vector = Float32MultiArray()
        vector.data = [
            float(control.get("steer") or 0.0),
            float(control.get("throttle") or 0.0),
            float(control.get("brake") or 0.0),
            float(infer_ms),
        ]
        self.pub_control.publish(vector)
        self._control_text = (
            f"step {control.get('step')}  "
            f"steer {vector.data[0]:+.2f}  thr {vector.data[1]:.2f}  "
            f"brk {vector.data[2]:.2f}  {infer_ms:.0f}ms"
        )

    def _on_sensor(self, message: UInt8MultiArray) -> None:
        frame = self.codec.decode(bytes(message.data))
        sensors = frame.get("sensors", {})
        stamp = rospy.Time.now()
        self._publish_lidar(sensors, stamp)
        self._publish_radar(sensors, stamp)
        self._publish_cameras(sensors, stamp)
        self._publish_pose(sensors, stamp)

    def _on_plan(self, message: UInt8MultiArray) -> None:
        plan = self.codec.decode(bytes(message.data)).get("plan") or {}
        stamp = rospy.Time.now()
        if "future_waypoints" in plan:
            self.pub_path.publish(
                path2(np.asarray(plan["future_waypoints"]), BASE_FRAME, stamp),
            )
        if "route" in plan:
            self.pub_route.publish(path2(np.asarray(plan["route"]), BASE_FRAME, stamp))
        self.pub_markers.publish(self._markers(plan, stamp))

    def _publish_lidar(self, sensors: dict, stamp: rospy.Time) -> None:
        clouds = []
        for index, key in enumerate(("lidar1", "lidar2")):
            raw = array_of(sensors.get(key))
            if raw is not None and raw.ndim == 2 and raw.shape[1] >= 4:
                clouds.append(
                    to_ego(
                        raw[:, :4].astype(np.float32),
                        LIDAR_POS[index],
                        LIDAR_ROT[index][2],
                    ),
                )
        if not clouds:
            return
        points = np.concatenate(clouds, axis=0)
        self.pub_lidar.publish(
            cloud2(points, ("x", "y", "z", "intensity"), BASE_FRAME, stamp),
        )

    def _publish_radar(self, sensors: dict, stamp: rospy.Time) -> None:
        points = []
        for index, key in enumerate(("radar1", "radar2", "radar3", "radar4")):
            raw = array_of(sensors.get(key))
            if raw is None or raw.ndim != 2 or raw.shape[1] < 4:
                continue
            distance, altitude, azimuth = raw[:, 0], raw[:, 1], raw[:, 2]
            cartesian = np.stack(
                [
                    distance * np.cos(azimuth) * np.cos(altitude),
                    distance * np.sin(azimuth) * np.cos(altitude),
                    distance * np.sin(altitude),
                ],
                axis=1,
            ).astype(np.float32)
            points.append(
                np.concatenate(
                    [
                        to_ego(cartesian, RADAR_POS[index], RADAR_ROT[index][2]),
                        raw[:, 3:4],
                    ],
                    axis=1,
                ),
            )
        if points:
            merged = np.concatenate(points, axis=0)
            self.pub_radar.publish(
                cloud2(merged, ("x", "y", "z", "velocity"), BASE_FRAME, stamp),
            )

    def _publish_cameras(self, sensors: dict, stamp: rospy.Time) -> None:
        images = []
        for key in sorted(sensors):
            if not key.startswith("rgb_"):
                continue
            image = array_of(sensors[key])
            if image is None or image.ndim != 3:
                continue
            index = int(key.split("_", 1)[1])
            self._camera_publisher(index).publish(
                image2(image, f"camera_{index}", stamp),
            )
            images.append(image[:, :, :3])
        if images:
            stitched = np.concatenate(images, axis=1)
            self.pub_camera.publish(image2(stitched, BASE_FRAME, stamp))

    def _publish_pose(self, sensors: dict, stamp: rospy.Time) -> None:
        gnss = array_of(sensors.get("gps"))
        if gnss is None:
            return
        self._gnss.append(tmerc_to_map(gnss, self._lat_ref, self._lon_ref))
        window = np.asarray(self._gnss)
        position = window.mean(axis=0)
        yaw = 0.0
        if len(window) >= 8:
            # Least-squares heading over the window: far steadier than one
            # frame delta when the fixes jitter by about a metre.
            steps = np.arange(len(window), dtype=np.float64)
            slope_x = float(np.polyfit(steps, window[:, 0], 1)[0])
            slope_y = float(np.polyfit(steps, window[:, 1], 1)[0])
            if float(np.hypot(slope_x, slope_y)) > 1e-6:
                yaw = math.atan2(slope_y, slope_x)
        self._broadcast_pose(position, yaw, stamp)
        self.pub_odom.publish(self._odometry(position, yaw, stamp))

    def _broadcast_pose(
        self,
        position: np.ndarray,
        yaw: float,
        stamp: rospy.Time,
    ) -> None:
        if self._tf is None:
            return
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = MAP_FRAME
        transform.child_frame_id = BASE_FRAME
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        self._tf.sendTransform(transform)

    def _odometry(
        self,
        position: np.ndarray,
        yaw: float,
        stamp: rospy.Time,
    ) -> Odometry:
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = MAP_FRAME
        message.child_frame_id = BASE_FRAME
        message.pose.pose.position.x = float(position[0])
        message.pose.pose.position.y = float(position[1])
        message.pose.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(yaw / 2.0)
        return message

    def _markers(self, plan: dict, stamp: rospy.Time) -> MarkerArray:
        markers = MarkerArray()
        markers.markers.append(self._ego_box(stamp))
        for index, (key, color) in enumerate(
            (
                ("previous_target_point", (0.0, 0.8, 0.0, 1.0)),
                ("target_point", (0.9, 0.1, 0.1, 1.0)),
                ("next_target_point", (1.0, 0.6, 0.0, 1.0)),
            ),
        ):
            point = plan.get(key)
            if point is not None:
                markers.markers.append(
                    self._sphere(index + 1, key, np.asarray(point), color, stamp),
                )
        markers.markers.append(self._text(4, stamp))
        return markers

    def _ego_box(self, stamp: rospy.Time) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = BASE_FRAME
        marker.ns = "ego"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.z = 0.8
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2 * EGO_EXTENT_X
        marker.scale.y = 2 * EGO_EXTENT_Y
        marker.scale.z = 1.6
        marker.color.r, marker.color.g, marker.color.b = (0.1, 0.9, 0.2)
        marker.color.a = 0.5
        return marker

    def _sphere(
        self,
        marker_id: int,
        name: str,
        point: np.ndarray,
        color: tuple[float, float, float, float],
        stamp: rospy.Time,
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = BASE_FRAME
        marker.ns = name
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.6
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def _text(self, marker_id: int, stamp: rospy.Time) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = BASE_FRAME
        marker.ns = "control"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 6.0
        marker.pose.position.z = 3.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 1.2
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.text = self._control_text
        return marker


def main() -> int:
    """Run the visualization node until shutdown.

    Returns:
        The process exit code.
    """
    rospy.init_node(os.environ.get("SIL_VIZ_NODE_NAME", "lead_sil_viz"))
    SilViz()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
