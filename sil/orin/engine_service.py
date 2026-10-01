#!/usr/bin/env python3
"""TensorRT engine service for the Orin SIL split (system Python 3.8).

This JetPack (L4T 35.1 / JetPack 5.0.2) ships TensorRT 8.4 Python bindings only
for the system Python 3.8, while ``lead`` needs Python 3.10. The py3.10 agent
therefore cannot load the ``.engine`` itself; this service owns the engine and
runs it, and the agent talks to it over a local ZeroMQ REQ/REP socket.

It mirrors ``model_quantization/scripts/benchmark_latency.py``: TensorRT 8.4's
binding API (``num_bindings``/``get_binding_*``/``execute_v2``) plus ``libcudart``
via ``ctypes`` for device buffers, so no CUDA-enabled torch is needed.

Protocol (msgpack via ``engine_codec``):

    request  {"v": 1, "feed": {<input name>: ndarray, ...}}
    reply    {"v": 1, "outputs": {<output name>: ndarray, ...}}
             {"v": 1, "error": "<message>"}

Run it under the system interpreter::

    LEAD_QUANTIZED_ENGINE=/path/to/model_fp16.engine \
        /usr/bin/python3 sil/orin/engine_service.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from ctypes import c_int, c_size_t, c_void_p
from pathlib import Path

import numpy as np
import tensorrt as trt
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_codec  # noqa: E402

ENGINE_PATH_ENV = "LEAD_QUANTIZED_ENGINE"
ENDPOINT_ENV = "SIL_ENGINE_ENDPOINT"
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5562"

_TRT_DTYPE = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT8: np.int8,
    trt.DataType.BOOL: np.bool_,
}


def _load_cudart() -> ctypes.CDLL:
    """Load libcudart and declare the calls the engine loop needs."""
    for name in (
        "libcudart.so",
        "libcudart.so.11.0",
        "/usr/local/cuda-11.4/lib64/libcudart.so.11.0",
    ):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise SystemExit("libcudart not found")
    lib.cudaMalloc.argtypes = [ctypes.POINTER(c_void_p), c_size_t]
    lib.cudaMalloc.restype = c_int
    lib.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
    lib.cudaMemcpy.restype = c_int
    return lib


class EngineService:
    """Load one engine and serve inference requests until interrupted."""

    def __init__(self, engine_path: str, endpoint: str) -> None:
        self.engine_path = engine_path
        self.endpoint = endpoint
        self.lib = _load_cudart()

        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise SystemExit("could not deserialise the engine at %s" % engine_path)
        self.context = self.engine.create_execution_context()

        self.buffers: dict[int, c_void_p] = {}
        self.host: dict[int, np.ndarray] = {}
        self.inputs: list[tuple[int, str]] = []
        self.outputs: list[tuple[int, str]] = []
        self._allocate()

        self.context_zmq = zmq.Context()
        self.socket = self.context_zmq.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)

    def _allocate(self) -> None:
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            shape = tuple(int(dim) for dim in self.engine.get_binding_shape(index))
            dtype = _TRT_DTYPE[self.engine.get_binding_dtype(index)]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            pointer = c_void_p()
            if self.lib.cudaMalloc(ctypes.byref(pointer), nbytes) != 0:
                raise SystemExit("cudaMalloc failed for %s" % name)
            self.buffers[index] = pointer
            self.host[index] = np.zeros(shape, dtype=dtype)
            if self.engine.binding_is_input(index):
                self.inputs.append((index, name))
            else:
                self.outputs.append((index, name))

    def _bindings(self) -> list[int]:
        return [self.buffers[i].value for i in range(self.engine.num_bindings)]

    def infer(self, feed: dict) -> dict:
        """Copy the feed in, execute, and copy the outputs back."""
        missing = [name for _, name in self.inputs if name not in feed]
        if missing:
            raise KeyError("the engine expects inputs %s" % missing)
        for index, name in self.inputs:
            host = self.host[index]
            source = np.ascontiguousarray(feed[name], dtype=host.dtype)
            if source.shape != host.shape:
                raise ValueError(
                    "input %s: expected %s, got %s" % (name, host.shape, source.shape)
                )
            host[...] = source
            if self.lib.cudaMemcpy(
                self.buffers[index], host.ctypes.data, host.nbytes, 1
            ) != 0:
                raise RuntimeError("cudaMemcpy H2D failed for %s" % name)
        if not self.context.execute_v2(self._bindings()):
            raise RuntimeError("execute_v2 failed")
        outputs = {}
        for index, name in self.outputs:
            host = self.host[index]
            if self.lib.cudaMemcpy(
                host.ctypes.data, self.buffers[index], host.nbytes, 2
            ) != 0:
                raise RuntimeError("cudaMemcpy D2H failed for %s" % name)
            outputs[name] = host.copy()
        return outputs

    def warmup(self, runs: int) -> None:
        feed = {name: self.host[index] for index, name in self.inputs}
        for _ in range(runs):
            self.infer(feed)

    def serve(self) -> None:
        print("ENGINE_SERVICE_READY %s" % self.endpoint, flush=True)
        while True:
            payload = self.socket.recv()
            started = time.perf_counter()
            try:
                request = engine_codec.decode(payload)
                outputs = self.infer(request["feed"])
                reply = {
                    "v": 1,
                    "outputs": outputs,
                    "infer_ms": (time.perf_counter() - started) * 1000.0,
                }
            except Exception as exc:  # keep serving after a bad frame
                reply = {"v": 1, "error": "%s: %s" % (type(exc).__name__, exc)}
            self.socket.send(engine_codec.encode(reply))


def main() -> int:
    parser = argparse.ArgumentParser(description="Orin TensorRT engine service")
    parser.add_argument(
        "--engine",
        default=os.environ.get(ENGINE_PATH_ENV, ""),
        help="Path to the .engine file (default: $%s)" % ENGINE_PATH_ENV,
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT),
        help="ZeroMQ REP endpoint to bind (default: %s)" % DEFAULT_ENDPOINT,
    )
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if not args.engine:
        parser.error("no engine: pass --engine or set %s" % ENGINE_PATH_ENV)

    service = EngineService(args.engine, args.endpoint)
    service.warmup(max(0, args.warmup))
    try:
        service.serve()
    except KeyboardInterrupt:
        print("engine service shutting down", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
