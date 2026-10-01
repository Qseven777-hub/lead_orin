"""Run a converted TensorRT engine in place of the policy's PyTorch forward.

This exists so a quantised model can be evaluated closed-loop with the ordinary
harness: select it with ``LEAD_CONFIG="policy.target=lead.policy.transfuser.quantized_policy:QuantizedTransfuser"``
and every evaluation script works unchanged.

It subclasses :class:`~lead.policy.transfuser.transfuser.Transfuser` and
overrides only ``forward``, so ``build_features``, ``features_to_batch``,
``input_cameras`` and the rest of the feature pipeline are *the same code* the
unquantised policy runs. That matters: any difference in the driving score then
comes from the quantised network, not from a reimplemented front end.

Two things about the evaluation harness shape this class:

* ``PolicyRunner`` calls ``load_state_dict`` with ``strict=True`` on the
  checkpoint's ``.pth``. The weights are already baked into the engine, so the
  call is swallowed -- there is nothing left to load.
* ``PolicyRunner`` also insists on exactly one ``model*.pth`` in the checkpoint
  directory. That is why the directory must keep the original checkpoint even
  though it goes unused.

The engine is built by ``model_quantization/export_lead_engine.py`` and its path
is taken from the ``LEAD_QUANTIZED_ENGINE`` environment variable. The engine's
outputs must be the four planning tensors (the driver reads nothing else), in
the order listed in :data:`OUTPUT_NAMES`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lead.config import LeadConfig
from lead.policy.transfuser.dataloader.sample import TransfuserForwardBatch
from lead.policy.transfuser.transfuser import Prediction, Transfuser

# Must match the order the engine was exported with. See
# model_quantization/lead_model.py, which is the source of truth for both lists.
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


def _import_tensorrt() -> Any:
    """Import the TensorRT bindings, preloading the libraries if needed.

    The bindings wheel ships no libraries; ``libnvinfer`` and its CUDA
    dependencies come from wherever TensorRT was unpacked. Setting
    ``LD_LIBRARY_PATH`` for the process is the usual way to make them findable,
    but that has to happen before the process starts. As a fallback, this
    preloads each library by absolute path with ``RTLD_GLOBAL``, which satisfies
    the same dependency without touching the environment.

    Returns:
        The ``tensorrt_bindings`` module.

    Raises:
        ImportError: If the bindings cannot be imported either way.
    """
    try:
        import tensorrt_bindings as trt  # type: ignore[reportMissingImports]

        return trt
    except ImportError:
        first_error = sys.exc_info()[1]

    prefix = Path(sys.prefix)
    candidates = [
        *sorted(prefix.glob("tensorrt-*/usr/lib/x86_64-linux-gnu/libnvonnxparser.so*")),
        *sorted(
            prefix.glob("tensorrt-*/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so*"),
        ),
        *sorted(prefix.glob("site-packages/nvidia/*/lib/*.so*")),
        *sorted(prefix.glob("lib/python3.*/site-packages/nvidia/*/lib/*.so*")),
        *sorted(prefix.glob("tensorrt-*/usr/lib/x86_64-linux-gnu/libnvinfer.so*")),
    ]
    for library in candidates:
        try:
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    try:
        import tensorrt_bindings as trt  # type: ignore[reportMissingImports]

        return trt
    except ImportError as exc:
        raise ImportError(
            "cannot import the TensorRT bindings. Install them with "
            "`pip install tensorrt-cu12-bindings==10.9.0.34`, and either set "
            "LD_LIBRARY_PATH to the unpacked TensorRT lib directory plus the "
            "environment's nvidia/*/lib directories before starting the "
            "process, or unpack TensorRT under the environment's prefix so the "
            f"preload fallback can find it. Original error: {first_error}",
        ) from exc


class TensorRTEngineRunner:
    """Loads one TRT engine and runs it on CUDA via torch-allocated buffers."""

    def __init__(self, engine_path: str | Path, device: int = 0) -> None:
        """Deserialise the engine and prepare one execution context.

        Args:
            engine_path: Path to the ``.engine`` file built by trtexec.
            device: CUDA device index to run on.
        """
        self.trt = _import_tensorrt()
        self.device = device
        blob = Path(engine_path).read_bytes()
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"could not deserialise the engine at {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = None
        self.buffers: dict[str, torch.Tensor] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self._allocate()

    def _allocate(self) -> None:
        """Allocate one CUDA buffer per binding and bind its address."""
        with torch.cuda.device(self.device):
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
                dtype = torch.from_numpy(
                    np.empty(
                        0,
                        dtype=self.trt.nptype(
                            self.engine.get_tensor_dtype(name),
                        ),
                    ),
                ).dtype
                tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{self.device}")
                self.buffers[name] = tensor
                if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                    self.inputs.append(name)
                else:
                    self.outputs.append(name)
                self.context.set_tensor_address(name, tensor.data_ptr())

    def infer(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one inference.

        Args:
            feed: One array per input name.

        Returns:
            One array per output name.

        Raises:
            KeyError: If an input the engine declares is missing from ``feed``.
        """
        missing = [name for name in self.inputs if name not in feed]
        if missing:
            raise KeyError(f"the engine expects inputs {missing}")
        for name in self.inputs:
            array = np.ascontiguousarray(feed[name], dtype=np.float32)
            self.buffers[name].copy_(
                torch.from_numpy(array).to(self.buffers[name].device),
            )
        if self.stream is None:
            self.stream = torch.cuda.current_stream(self.device).cuda_stream
        self.context.execute_async_v3(stream_handle=self.stream)
        torch.cuda.synchronize(self.device)
        return {name: self.buffers[name].cpu().numpy() for name in self.outputs}


class QuantizedTransfuser(Transfuser):
    """The Transfuser policy, computed by a TensorRT engine instead of PyTorch.

    Only ``forward`` differs from the parent; everything the evaluation harness
    touches -- feature building, the policy config, the camera list -- is
    inherited unchanged.

    The engine path comes from the ``LEAD_QUANTIZED_ENGINE`` environment
    variable.
    """

    def __init__(self, lead_config: LeadConfig) -> None:
        """Build the parent policy and load the engine.

        Args:
            lead_config: The resolved config tree.

        Raises:
            RuntimeError: If ``LEAD_QUANTIZED_ENGINE`` is unset.
        """
        super().__init__(lead_config)
        engine_path = os.environ.get(ENGINE_PATH_ENV)
        if not engine_path:
            raise RuntimeError(
                f"set {ENGINE_PATH_ENV} to the .engine file to evaluate "
                "(it is built by model_quantization/export_lead_engine.py)",
            )
        self.engine_path = engine_path
        self.runner = TensorRTEngineRunner(engine_path)

    def load_state_dict(
        self,
        *args: object,
        **kwargs: object,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Accept the checkpoint without loading it.

        The weights are compiled into the engine, so there is nothing to load
        here. Swallowing the call keeps ``PolicyRunner``'s ``strict=True`` load
        from failing on a model that legitimately has no parameters to fill.

        Args:
            *args: Positional arguments, ignored; matches the parent module.
            **kwargs: Keyword arguments, ignored; matches the parent module.

        Returns:
            An empty ``_IncompatibleKeys``, as ``nn.Module`` would return.
        """
        return torch.nn.modules.module._IncompatibleKeys([], [])  # noqa: SLF001

    def forward(self, batch: TransfuserForwardBatch) -> Prediction:
        """Run the engine and wrap its outputs as a ``Prediction``.

        Args:
            batch: The batched model inputs built by ``features_to_batch``.

        Returns:
            A prediction carrying the four planning outputs; the auxiliary
            heads are not in this engine, so they stay ``None``.
        """
        feed = {}
        for name in INPUT_NAMES:
            tensor = batch[name]
            feed[name] = np.ascontiguousarray(
                tensor.detach().cpu().numpy(),
                dtype=np.float32,
            )
        results = self.runner.infer(feed)
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
