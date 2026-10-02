"""Standalone msgpack + ndarray codec for the Orin engine service link.

The py3.8 TensorRT service cannot import the shared ``lead`` package (it needs
Python 3.10), so this small module carries the exact wire format the SIL codec
uses -- msgpack with a ``__ndarray__`` marker -- and is importable from both the
3.8 service and the 3.10 compute process.

Only ``None``/``bool``/``int``/``float``/``str``/``bytes``/list/dict and numpy
arrays are supported, which is all the engine feed/reply needs.
"""

from __future__ import annotations

import typing

import msgpack
import numpy as np

_NDARRAY_MARKER = "__ndarray__"


def _to_wire(value: typing.Any) -> typing.Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            _NDARRAY_MARKER: True,
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "data": contiguous.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(
        f"cannot encode {type(value).__name__} for the engine wire format"
    )


def _from_wire(value: typing.Any) -> typing.Any:
    if isinstance(value, dict):
        if value.get(_NDARRAY_MARKER) is True:
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            # frombuffer shares the msgpack buffer; copy so the caller keeps it.
            return array.reshape(value["shape"]).copy()
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    return value


def encode(message: dict) -> bytes:
    """Encode a dict (with ndarrays) into wire bytes."""
    return msgpack.packb(_to_wire(message), use_bin_type=True)


def decode(payload: bytes) -> dict:
    """Decode wire bytes back into a dict."""
    return _from_wire(msgpack.unpackb(payload, raw=False))
