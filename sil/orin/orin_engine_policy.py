"""Orin engine-backed Transfuser policy (py3.10 side of the SIL split).

Mirrors ``lead.policy.transfuser.quantized_policy.QuantizedTransfuser`` on
purpose -- only ``forward`` differs from the parent ``Transfuser``, and
``load_state_dict`` is swallowed because the weights live in the engine.  The
difference is *where* the engine runs: this Orin cannot import the TensorRT 8.4
bindings from Python 3.10, so the engine lives in ``sil/orin/engine_service.py``
(system Python 3.8) and this policy talks to it over a local ZeroMQ REQ/REP
socket.  Everything else -- ``build_features``, ``features_to_batch``,
``input_cameras`` -- is inherited unchanged, so the front end is byte-for-byte
the shared one.

Select it with::

    LEAD_CONFIG="policy.target=orin_engine_policy:OrinEngineTransfuser"
    LEAD_QUANTIZED_ENGINE=/path/to/model_fp16.engine
"""

from __future__ import annotations

import os
import sys
import typing
from pathlib import Path

import numpy as np
import torch
import zmq

from lead.config import LeadConfig
from lead.policy.transfuser.transfuser import Prediction, Transfuser

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_codec  # noqa: E402

# Must match the order the engine was exported with.  See
# model_quantization/lead_model.py, the source of truth for both lists.
INPUT_NAMES: tuple[str, ...] = (
    "rgb",
    "rasterized_lidar",
    "radar",
    "previous_target_point",
    "target_point",
    "next_target_point",
    "speed",
)

OUTPUT_NAMES: tuple[str, ...] = (
    "future_waypoints",
    "target_speed_scalar",
    "target_speed_distribution",
    "route",
)

ENGINE_PATH_ENV = "LEAD_QUANTIZED_ENGINE"
ENDPOINT_ENV = "SIL_ENGINE_ENDPOINT"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5562"
REQUEST_TIMEOUT_MS = 10_000


class OrinEngineTransfuser(Transfuser):
    """The Transfuser policy, computed by the py3.8 TensorRT engine service."""

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the parent policy and connect to the engine service.

        Args:
            lead_config: The resolved config tree.

        Raises:
            RuntimeError: If ``LEAD_QUANTIZED_ENGINE`` is unset.
        """
        super().__init__(lead_config)
        engine_path = os.environ.get(ENGINE_PATH_ENV)
        if not engine_path:
            raise RuntimeError(
                f"set {ENGINE_PATH_ENV} to the .engine file the service loads",
            )
        self.engine_path = engine_path
        self.endpoint = os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT)
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, REQUEST_TIMEOUT_MS)
        self._socket.connect(self.endpoint)

    def load_state_dict(
        self,
        *args: object,
        **kwargs: object,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Accept the checkpoint without loading it.

        The weights are compiled into the engine, so there is nothing to load
        here. Swallowing the call keeps ``PolicyRunner``'s strict load from
        failing on a model that legitimately has no parameters to fill.
        """
        return torch.nn.modules.module._IncompatibleKeys([], [])  # noqa: SLF001

    def forward(self, batch: typing.Mapping[str, typing.Any]) -> Prediction:
        """Send the batch to the engine service and wrap its outputs.

        Args:
            batch: The batched model inputs built by ``features_to_batch``.

        Returns:
            A prediction carrying the four planning outputs; the auxiliary
            heads are not in this engine, so they stay ``None``.

        Raises:
            RuntimeError: If the service errors, times out, or omits an output.
        """
        feed = {}
        for name in INPUT_NAMES:
            tensor = batch[name]
            feed[name] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(),
                dtype=np.float32,
            )
        self._socket.send(engine_codec.encode({"v": 1, "feed": feed}))
        try:
            reply = engine_codec.decode(self._socket.recv())
        except zmq.Again as exc:
            raise RuntimeError(
                f"engine service at {self.endpoint} did not reply within "
                f"{REQUEST_TIMEOUT_MS} ms",
            ) from exc
        if "error" in reply:
            raise RuntimeError(f"engine service error: {reply['error']}")
        results = reply["outputs"]
        missing = [name for name in OUTPUT_NAMES if name not in results]
        if missing:
            raise RuntimeError(f"the engine did not return outputs {missing}")
        device = next(self.parameters()).device
        tensors = {
            name: torch.from_numpy(results[name]).to(device) for name in OUTPUT_NAMES
        }
        return Prediction(
            future_waypoints=tensors["future_waypoints"],
            target_speed_scalar=tensors["target_speed_scalar"],
            target_speed_distribution=tensors["target_speed_distribution"],
            route=tensors["route"],
            semantic=None,
            bev_semantic=None,
            depth=None,
            bounding_box=None,
            radar_features=None,
            radar_predictions=None,
        )
