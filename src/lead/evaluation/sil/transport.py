"""ZeroMQ transport between a Python 3.10 compute process and its ROS bridge.

The ROS1 transport runs in a Python 3.8 container (see ``sil/ros_bridge``).
A compute process reaches it over two local ZeroMQ PUSH/PULL sockets: the
outgoing socket carries ``[topic, payload]`` frames to be published on ROS,
the incoming socket carries ``[topic, payload]`` frames the bridge received.

PUSH/PULL is used rather than PUB/SUB so a message queued before the bridge is
connected is not dropped.

ZeroMQ sockets are not thread-safe. A compute process can send from more than
one thread (the Orin node sends controls from its main loop and heartbeats from
a timer thread), so sends and receives are each serialized with a lock.
"""

from __future__ import annotations

import threading
import time

import zmq

DEFAULT_OUT_ENDPOINT = "tcp://127.0.0.1:5560"
DEFAULT_IN_ENDPOINT = "tcp://127.0.0.1:5561"


class SilTransport:
    """Publish and receive SIL messages through the local ROS bridge."""

    def __init__(
        self,
        out_endpoint: str = DEFAULT_OUT_ENDPOINT,
        in_endpoint: str = DEFAULT_IN_ENDPOINT,
        *,
        context: zmq.Context | None = None,
        linger_ms: int = 0,
    ) -> None:
        """Connect the outgoing and incoming sockets.

        Args:
            out_endpoint: Endpoint the bridge PULLs from (we connect and send).
            in_endpoint: Endpoint the bridge PUSHes to (we connect and receive).
            context: Optional shared ZeroMQ context.
            linger_ms: Socket linger on close; zero drops pending messages.
        """
        self._context = context or zmq.Context.instance()
        self._sink = self._context.socket(zmq.PUSH)
        self._sink.setsockopt(zmq.LINGER, linger_ms)
        self._sink.connect(out_endpoint)
        self._source = self._context.socket(zmq.PULL)
        self._source.setsockopt(zmq.LINGER, linger_ms)
        self._source.connect(in_endpoint)
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send(self, topic: str, payload: bytes) -> None:
        """Queue one message for the bridge to publish.

        Args:
            topic: ROS topic name.
            payload: Encoded message bytes.
        """
        with self._send_lock:
            self._sink.send_multipart([topic.encode("utf-8"), payload])

    def recv(self, timeout_ms: int) -> tuple[str, bytes] | None:
        """Receive one message with a timeout.

        Args:
            timeout_ms: How long to wait, in milliseconds.

        Returns:
            The ``(topic, payload)`` pair, or ``None`` on timeout.
        """
        with self._recv_lock:
            if self._source.poll(timeout_ms) == 0:
                return None
            topic, payload = self._source.recv_multipart()
        return topic.decode("utf-8"), payload

    def wait_for(self, topic: str, timeout_ms: int) -> bytes | None:
        """Wait for the next message on one topic, dropping the others.

        Args:
            topic: Topic to wait for.
            timeout_ms: Overall deadline in milliseconds.

        Returns:
            The payload, or ``None`` if the deadline passed first.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000.0)
            if remaining_ms <= 0:
                return None
            item = self.recv(remaining_ms)
            if item is None:
                return None
            received_topic, payload = item
            if received_topic == topic:
                return payload

    def close(self) -> None:
        """Close both sockets; the context is left to its owner."""
        self._sink.close()
        self._source.close()
