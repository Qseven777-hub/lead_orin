"""Wire contract between the CARLA host and the Orin compute process.

Every message is a plain ``dict`` that :mod:`lead.evaluation.sil.codec` turns
into bytes. All topics carry one ``kind`` and the same protocol version, so a
receiver can reject a mismatched peer early.

Topics
------
- ``lead/session``     host -> Orin, once per route (route metadata and the
  navigation plan the Orin cannot derive without CARLA).
- ``lead/sensor_frame`` host -> Orin, once per tick (the raw leaderboard sensor
  frame).
- ``lead/control``     Orin -> host, once per tick (the vehicle control).
- ``lead/heartbeat``   Orin -> host, periodic (liveness and timing).
- ``lead/error``       either direction, on failure.
"""

from __future__ import annotations

import typing

PROTOCOL_VERSION = 1

TOPIC_SESSION = "lead/session"
TOPIC_SENSOR = "lead/sensor_frame"
TOPIC_CONTROL = "lead/control"
TOPIC_HEARTBEAT = "lead/heartbeat"
TOPIC_ERROR = "lead/error"

# Topics the host compute process expects back from the Orin.
ORIN_TO_HOST_TOPICS: tuple[str, ...] = (
    TOPIC_CONTROL,
    TOPIC_HEARTBEAT,
    TOPIC_ERROR,
)

# Non-camera leaderboard sensor ids the agent's ``tick`` reads.
SENSOR_KEYS: tuple[str, ...] = (
    "lidar1",
    "lidar2",
    "radar1",
    "radar2",
    "radar3",
    "radar4",
    "gps",
    "imu",
    "speed",
)


def camera_key(index: int) -> str:
    """Leaderboard sensor id of camera ``index`` (1-based).

    Args:
        index: 1-based camera index in the sensor rig.

    Returns:
        The ``rgb_<index>`` id the leaderboard uses.
    """
    return f"rgb_{index}"


def _envelope(kind: str) -> dict[str, typing.Any]:
    return {"v": PROTOCOL_VERSION, "kind": kind}


def session(
    *,
    route_id: str,
    session_id: str,
    scenario_type: str,
    map_name: str,
    gnss_uses_transverse_mercator: bool,
    global_plan_gps: list[dict[str, typing.Any]],
    lat_ref: float,
    lon_ref: float,
    camera_indices: tuple[int, ...],
    config_source: str,
) -> dict[str, typing.Any]:
    """Build the once-per-route session message.

    Args:
        route_id: Leaderboard route id (``BENCHMARK_ROUTE_ID``).
        session_id: Unique id of this route run. The host keeps it stable across
            re-sends of the same session, and changes it for a new run of the
            same route, so the Orin can tell a duplicate from a fresh start.
        scenario_type: Scenario type parsed from the routes XML.
        map_name: CARLA map name, replacing ``self._world.get_map().name``.
        gnss_uses_transverse_mercator: GNSS projection flag the agent otherwise
            reads from ``CarlaDataProvider.get_client()``.
        global_plan_gps: Leaderboard global plan as ``{lat, lon, z?, command}``
            dicts.
        lat_ref: GPS latitude reference, computed on the host from the world plan.
        lon_ref: GPS longitude reference, computed on the host from the world plan.
        camera_indices: Leaderboard camera indices the host forwards and the core
            should preprocess.
        config_source: Human-readable origin of the lead config the Orin must
            rebuild (checkpoint directory or override string), for logging only.

    Returns:
        The session message payload.
    """
    message = _envelope("session")
    message.update(
        {
            "route_id": route_id,
            "session_id": session_id,
            "scenario_type": scenario_type,
            "map_name": map_name,
            "gnss_uses_transverse_mercator": gnss_uses_transverse_mercator,
            "global_plan_gps": global_plan_gps,
            "lat_ref": lat_ref,
            "lon_ref": lon_ref,
            "camera_indices": list(camera_indices),
            "config_source": config_source,
        },
    )
    return message


def sensor_frame(
    *,
    seq: int,
    step: int,
    sim_time_us: int,
    sensors: dict[str, typing.Any],
    camera_indices: tuple[int, ...],
) -> dict[str, typing.Any]:
    """Build one tick's sensor frame.

    Args:
        seq: Monotonic frame sequence, echoed back on the control message.
        step: Agent step counter (``self.step``).
        sim_time_us: Simulation timestamp in microseconds.
        sensors: Raw per-sensor payloads keyed by leaderboard sensor id.
        camera_indices: Camera indices whose ``rgb_<i>`` images are included.

    Returns:
        The sensor frame message payload.
    """
    message = _envelope("sensor_frame")
    message.update(
        {
            "seq": seq,
            "step": step,
            "sim_time_us": sim_time_us,
            "sensors": sensors,
            "camera_indices": list(camera_indices),
        },
    )
    return message


def control(
    *,
    seq: int,
    step: int,
    steer: float,
    throttle: float,
    brake: float,
    infer_ms: float | None = None,
    aux: dict[str, typing.Any] | None = None,
) -> dict[str, typing.Any]:
    """Build the control reply for one sensor frame.

    Args:
        seq: Sequence of the sensor frame being answered.
        step: Agent step counter the Orin reached.
        steer: Steering control in ``[-1, 1]``.
        throttle: Throttle control in ``[0, 1]``.
        brake: Brake control in ``[0, 1]``.
        infer_ms: Optional Orin-side inference time in milliseconds.
        aux: Optional extra tensors (e.g. predictions for visualization).

    Returns:
        The control message payload.
    """
    message = _envelope("control")
    message.update(
        {
            "seq": seq,
            "step": step,
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
            "infer_ms": infer_ms,
            "aux": aux,
        },
    )
    return message


def heartbeat(
    *,
    seq: int,
    status: str,
    engine_ready: bool,
    last_infer_ms: float | None = None,
    gpu_mem_mb: float | None = None,
) -> dict[str, typing.Any]:
    """Build a liveness message from the Orin.

    Args:
        seq: Monotonic heartbeat sequence.
        status: Free-form status string.
        engine_ready: Whether the engine is loaded and serving.
        last_infer_ms: Most recent inference time in milliseconds.
        gpu_mem_mb: Optional GPU memory usage in MiB.

    Returns:
        The heartbeat message payload.
    """
    message = _envelope("heartbeat")
    message.update(
        {
            "seq": seq,
            "status": status,
            "engine_ready": engine_ready,
            "last_infer_ms": last_infer_ms,
            "gpu_mem_mb": gpu_mem_mb,
        },
    )
    return message


def error(
    *,
    code: str,
    message: str,
    seq: int | None = None,
) -> dict[str, typing.Any]:
    """Build an error message.

    Args:
        code: Short machine-readable error code.
        message: Human-readable description.
        seq: Sequence the error relates to, if any.

    Returns:
        The error message payload.
    """
    payload = _envelope("error")
    payload.update({"code": code, "message": message, "seq": seq})
    return payload
