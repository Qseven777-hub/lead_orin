#!/usr/bin/env python3
"""ROS1 <-> ZeroMQ bridge for the SIL split.

Runs beside roscore in a Python 3.8/3.9 ROS Noetic environment (RoboStack conda
env or the container). On one side it speaks ROS1: every frame arriving from the
compute process is published on its topic as a ``std_msgs/UInt8MultiArray``, and
every subscribed ROS topic is forwarded back to the compute process. On the
other side it speaks two local ZeroMQ PUSH/PULL sockets, so the Python 3.10
compute process never has to import rospy.

Environment:
    SIL_BRIDGE_IN         PULL endpoint bound for compute -> ROS (default 5560)
    SIL_BRIDGE_OUT        PUSH endpoint bound for ROS -> compute (default 5561)
    SIL_BRIDGE_SUBSCRIBE  comma-separated ROS topics to forward to compute
    SIL_BRIDGE_NODE_NAME  ROS node name (default lead_sil_bridge)
"""

from __future__ import annotations

import os
import threading

import rospy
import zmq
from std_msgs.msg import UInt8MultiArray

DEFAULT_SUBSCRIBE = (
    "lead/control,lead/heartbeat,lead/error,lead/session,lead/sensor_frame"
)
POLL_TIMEOUT_MS = 200
QUEUE_SIZE = 10


class RosBridge:
    """Forward between local ZeroMQ frames and ROS ``UInt8MultiArray`` topics."""

    def __init__(self) -> None:
        in_port = os.environ.get("SIL_BRIDGE_IN", "5560")
        out_port = os.environ.get("SIL_BRIDGE_OUT", "5561")
        self._pull_endpoint = f"tcp://127.0.0.1:{in_port}"
        self._push_endpoint = f"tcp://127.0.0.1:{out_port}"
        self._subscribe = [
            topic.strip()
            for topic in os.environ.get(
                "SIL_BRIDGE_SUBSCRIBE", DEFAULT_SUBSCRIBE
            ).split(",")
            if topic.strip()
        ]

        self._context = zmq.Context()
        self._pull = self._context.socket(zmq.PULL)
        self._pull.setsockopt(zmq.LINGER, 0)
        self._pull.bind(self._pull_endpoint)
        self._push = self._context.socket(zmq.PUSH)
        self._push.setsockopt(zmq.LINGER, 0)
        self._push.bind(self._push_endpoint)

        self._publishers: dict[str, rospy.Publisher] = {}
        self._publisher_lock = threading.Lock()
        # A ZeroMQ socket must not be used from two threads at once; rospy may
        # deliver callbacks from more than one.
        self._push_lock = threading.Lock()

    def _publisher(self, topic: str) -> rospy.Publisher:
        with self._publisher_lock:
            publisher = self._publishers.get(topic)
            if publisher is None:
                publisher = rospy.Publisher(
                    topic,
                    UInt8MultiArray,
                    queue_size=QUEUE_SIZE,
                )
                self._publishers[topic] = publisher
            return publisher

    def _forward_to_compute(self, message: UInt8MultiArray, topic: str) -> None:
        payload = bytes(message.data)
        with self._push_lock:
            self._push.send_multipart([topic.encode("utf-8"), payload])

    def _subscribe_all(self) -> None:
        for topic in self._subscribe:
            rospy.Subscriber(
                topic,
                UInt8MultiArray,
                self._forward_to_compute,
                callback_args=topic,
                queue_size=QUEUE_SIZE,
            )
            rospy.loginfo("bridge forwarding ROS topic %s -> compute", topic)

    def _compute_to_ros_loop(self) -> None:
        while not rospy.is_shutdown():
            if self._pull.poll(POLL_TIMEOUT_MS) == 0:
                continue
            topic_bytes, payload = self._pull.recv_multipart()
            topic = topic_bytes.decode("utf-8")
            message = UInt8MultiArray()
            # rospy stores uint8[] as bytes on Python 3; this avoids the
            # per-byte list conversion a multi-megabyte frame would otherwise pay.
            message.data = payload
            self._publisher(topic).publish(message)

    def run(self) -> None:
        """Start the node, subscribe, and pump compute frames until shutdown."""
        rospy.init_node(
            os.environ.get("SIL_BRIDGE_NODE_NAME", "lead_sil_bridge"),
            anonymous=False,
        )
        rospy.loginfo(
            "bridge PULL %s, PUSH %s", self._pull_endpoint, self._push_endpoint
        )
        self._subscribe_all()
        pump = threading.Thread(target=self._compute_to_ros_loop, daemon=True)
        pump.start()
        rospy.spin()
        self._pull.close()
        self._push.close()
        self._context.term()


def main() -> None:
    RosBridge().run()


if __name__ == "__main__":
    main()
