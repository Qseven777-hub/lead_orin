#!/usr/bin/env python3
"""Temporary Orin stand-in for local SIL loopback tests.

Subscribes to ``lead/sensor_frame``, echoes a fixed control on ``lead/control``
and a heartbeat, so the host compute path can be exercised end to end without
the Orin. It needs only msgpack and rospy.

Environment:
    SIL_STUB_STEER / SIL_STUB_THROTTLE / SIL_STUB_BRAKE  control to echo
"""

from __future__ import annotations

import os
import threading

import msgpack
import rospy
from std_msgs.msg import UInt8MultiArray

CONTROL_TOPIC = "lead/control"
HEARTBEAT_TOPIC = "lead/heartbeat"
SENSOR_TOPIC = "lead/sensor_frame"


def _pack(message: dict) -> bytes:
    return msgpack.packb(message, use_bin_type=True)


class StubOrin:
    """Echo the host sensor frames back as controls."""

    def __init__(self) -> None:
        self._steer = float(os.environ.get("SIL_STUB_STEER", "0.1"))
        self._throttle = float(os.environ.get("SIL_STUB_THROTTLE", "0.2"))
        self._brake = float(os.environ.get("SIL_STUB_BRAKE", "0.0"))
        self._control_publisher = rospy.Publisher(
            CONTROL_TOPIC,
            UInt8MultiArray,
            queue_size=10,
        )
        self._heartbeat_publisher = rospy.Publisher(
            HEARTBEAT_TOPIC,
            UInt8MultiArray,
            queue_size=10,
        )
        rospy.Subscriber(
            SENSOR_TOPIC,
            UInt8MultiArray,
            self._on_sensor_frame,
            queue_size=10,
        )
        self._heartbeat_seq = 0

    def _on_sensor_frame(self, message: UInt8MultiArray) -> None:
        frame = msgpack.unpackb(bytes(message.data), raw=False)
        control = {
            "v": 1,
            "kind": "control",
            "seq": frame["seq"],
            "step": frame["step"],
            "steer": self._steer,
            "throttle": self._throttle,
            "brake": self._brake,
            "infer_ms": 1.0,
            "aux": None,
        }
        output = UInt8MultiArray()
        output.data = _pack(control)
        self._control_publisher.publish(output)
        rospy.loginfo("stub replied control seq=%s", frame["seq"])

    def _heartbeat_loop(self) -> None:
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self._heartbeat_seq += 1
            heartbeat = {
                "v": 1,
                "kind": "heartbeat",
                "seq": self._heartbeat_seq,
                "status": "stub",
                "engine_ready": True,
                "last_infer_ms": 1.0,
                "gpu_mem_mb": None,
            }
            output = UInt8MultiArray()
            output.data = _pack(heartbeat)
            self._heartbeat_publisher.publish(output)
            rate.sleep()

    def run(self) -> None:
        rospy.init_node("lead_sil_stub_orin", anonymous=False)
        rospy.loginfo("stub Orin ready")
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        rospy.spin()


def main() -> None:
    StubOrin().run()


if __name__ == "__main__":
    main()
