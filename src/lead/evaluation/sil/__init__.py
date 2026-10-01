"""Software-in-the-loop (SIL) split between the CARLA host and the Orin.

The CARLA/leaderboard process runs on the host; the driving agent's
computation (localization, scene building, featurization, network forward,
control tracking) runs on the Orin. The two sides exchange the raw sensor
frame and the resulting control over ROS1, with a small byte transport that
keeps the ROS1/Python 3.8 transport container decoupled from the Python 3.10
compute processes.

This package holds the Python 3.10 side: the wire contract, the binary codec
and the ZeroMQ transport that a compute process uses to reach its local bridge.
"""

from lead.evaluation.sil.codec import decode, encode
from lead.evaluation.sil.contract import (
    PROTOCOL_VERSION,
    TOPIC_CONTROL,
    TOPIC_ERROR,
    TOPIC_HEARTBEAT,
    TOPIC_SENSOR,
    TOPIC_SESSION,
    control,
    error,
    heartbeat,
    sensor_frame,
    session,
)
from lead.evaluation.sil.transport import SilTransport

__all__ = [
    "PROTOCOL_VERSION",
    "TOPIC_CONTROL",
    "TOPIC_ERROR",
    "TOPIC_HEARTBEAT",
    "TOPIC_SENSOR",
    "TOPIC_SESSION",
    "SilTransport",
    "control",
    "decode",
    "encode",
    "error",
    "heartbeat",
    "sensor_frame",
    "session",
]
