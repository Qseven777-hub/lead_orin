#!/usr/bin/env python3
"""Orin-side SIL agent node.

Runs the CARLA-free TransFuser core behind the SIL transport: it receives the
once-per-route session and the per-tick sensor frames over ZeroMQ (via the local
ROS bridge), runs the policy, and returns the vehicle control. The policy is the
quantized TensorRT engine, selected through ``LEAD_QUANTIZED_ENGINE`` exactly as
a local evaluation would.

Run it in the Orin's Python 3.10 ``lead`` environment:

    LEAD_QUANTIZED_ENGINE=/path/to/model_fp16.engine \
    LEAD_CONFIG="policy.target=lead.policy.transfuser.quantized_policy:QuantizedTransfuser" \
    python sil/orin/agent_node.py --checkpoint /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

import torch
import yaml

from lead.common.logging_setup import setup_logging
from lead.config import load_lead_config
from lead.evaluation.agents.transfuser.transfuser_core import TransfuserCore
from lead.evaluation.sil import codec, contract
from lead.evaluation.sil.transport import SilTransport

# Orin-local sibling module; ``sil/orin/**`` is preserved by the sync script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from orin_policy_runner import OrinPolicyRunner  # noqa: E402

setup_logging()
LOG = logging.getLogger("lead_sil_orin")

HEARTBEAT_PERIOD_S = 0.5
RECV_TIMEOUT_MS = 1000


def load_checkpoint_config(checkpoint_dir: str):
    """Load the checkpoint's stored config tree, as the CARLA agent does."""
    with open(
        os.path.join(checkpoint_dir, "config.yaml"),
        encoding="utf-8",
    ) as handle:
        stored_config = yaml.safe_load(handle)
    stored_config.pop("evaluation", None)
    return load_lead_config(loaded_config=stored_config, raise_on_unknown_key=False)


class OrinAgentNode:
    """Serve the TransFuser core over the SIL transport."""

    def __init__(
        self,
        checkpoint_dir: str,
        device: torch.device,
        transport: SilTransport,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.transport = transport
        self.lead_config = load_checkpoint_config(checkpoint_dir)
        LOG.info("policy target: %s", self.lead_config.policy.target)
        self.runner = OrinPolicyRunner(
            lead_config=self.lead_config,
            model_path=checkpoint_dir,
            device=device,
        )
        self.core: TransfuserCore | None = None
        self._route_id: str | None = None
        self._stop = threading.Event()
        self._controls_sent = 0

    def _build_core(self, session: dict) -> None:
        global_plan = [
            (
                {
                    "lat": entry["lat"],
                    "lon": entry["lon"],
                    "z": float(entry.get("z", 0.0)),
                },
                entry.get("command"),
            )
            for entry in session["global_plan_gps"]
        ]
        self.core = TransfuserCore()
        self.core.setup_core(
            lead_config=self.lead_config,
            policy=self.runner.policy,
            forward=self.runner.forward,
            global_plan=global_plan,
            global_plan_world_coord=None,
            gnss_uses_transverse_mercator=bool(
                session["gnss_uses_transverse_mercator"],
            ),
            map_name=str(session["map_name"]),
            device=self.device,
            lat_ref=float(session["lat_ref"]),
            lon_ref=float(session["lon_ref"]),
            camera_indices=tuple(session.get("camera_indices", ())),
        )
        self._route_id = session.get("route_id")
        LOG.info(
            "core ready: route=%s map=%s",
            session.get("route_id"),
            session.get("map_name"),
        )

    def _on_sensor_frame(self, frame: dict) -> None:
        if self.core is None:
            self.transport.send(
                contract.TOPIC_ERROR,
                codec.encode(
                    contract.error(
                        code="no_session",
                        message="sensor_frame arrived before session",
                        seq=frame.get("seq"),
                    ),
                ),
            )
            return
        started = time.perf_counter()
        # Keep the core's step identical to the host's, so its timestamps and
        # temporal window line up with a local run.
        self.core.step = int(frame["step"]) - 1
        command = self.core.run_step(frame["sensors"])
        infer_ms = (time.perf_counter() - started) * 1000.0
        self.transport.send(
            contract.TOPIC_CONTROL,
            codec.encode(
                contract.control(
                    seq=int(frame["seq"]),
                    step=int(frame["step"]),
                    steer=command.steer,
                    throttle=command.throttle,
                    brake=command.brake,
                    infer_ms=infer_ms,
                ),
            ),
        )
        self._controls_sent += 1
        if self._controls_sent == 1:
            LOG.info(
                "first control sent: seq=%s steer=%.3f throttle=%.3f brake=%.3f infer_ms=%.1f",
                int(frame["seq"]),
                command.steer,
                command.throttle,
                command.brake,
                infer_ms,
            )

    def _heartbeat_loop(self) -> None:
        seq = 0
        while not self._stop.is_set():
            seq += 1
            self.transport.send(
                contract.TOPIC_HEARTBEAT,
                codec.encode(
                    contract.heartbeat(
                        seq=seq,
                        status="ready" if self.core is not None else "waiting-session",
                        engine_ready=True,
                    ),
                ),
            )
            self._stop.wait(HEARTBEAT_PERIOD_S)

    def run(self) -> None:
        """Serve frames until interrupted."""
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        LOG.info("Orin SIL agent node listening")
        try:
            while True:
                item = self.transport.recv(RECV_TIMEOUT_MS)
                if item is None:
                    continue
                topic, payload = item
                if topic == contract.TOPIC_SESSION:
                    session = codec.decode(payload)
                    # The host re-sends the session on timeouts (ROS topics are
                    # not latched); only rebuild when the route actually changes,
                    # so a duplicate does not reset the core mid-route.
                    if self.core is None or session.get("route_id") != self._route_id:
                        self._build_core(session)
                    else:
                        LOG.debug(
                            "ignoring duplicate session for route %s",
                            self._route_id,
                        )
                elif topic == contract.TOPIC_SENSOR:
                    self._on_sensor_frame(codec.decode(payload))
                else:
                    LOG.debug("ignoring topic %s", topic)
        except KeyboardInterrupt:
            LOG.info("shutting down")
        finally:
            self._stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Orin SIL agent node")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory holding config.yaml and model*.pth",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument(
        "--out-endpoint",
        default=os.environ.get(
            "SIL_COMPUTE_OUT",
            "tcp://127.0.0.1:5560",
        ),
        help="ZeroMQ endpoint to the local ROS bridge (compute -> ROS)",
    )
    parser.add_argument(
        "--in-endpoint",
        default=os.environ.get(
            "SIL_COMPUTE_IN",
            "tcp://127.0.0.1:5561",
        ),
        help="ZeroMQ endpoint from the local ROS bridge (ROS -> compute)",
    )
    args = parser.parse_args()

    transport = SilTransport(args.out_endpoint, args.in_endpoint)
    node = OrinAgentNode(
        checkpoint_dir=args.checkpoint,
        device=torch.device(args.device),
        transport=transport,
    )
    node.run()


if __name__ == "__main__":
    main()
