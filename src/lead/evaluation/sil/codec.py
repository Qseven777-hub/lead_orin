"""Binary codec for the SIL wire format.

The transport carries opaque byte payloads between a compute process and its
local ROS bridge. This module turns the message dicts of
:mod:`lead.evaluation.sil.contract` into those bytes and back. The encoding is
msgpack with a small ndarray extension, which keeps the multi-megabyte sensor
frames compact and is identical on both Python 3.10 processes.

Only the types the contract uses are supported: ``None``, ``bool``, ``int``,
``float``, ``str``, ``bytes``, ``list``, ``tuple``, ``dict`` and numpy arrays
(any dtype and shape). Tuples keep their type across the round trip, because
the shared code type-checks some payloads as tuples. Anything else raises
:class:`TypeError`.
"""

from __future__ import annotations

import typing

import msgpack
import numpy as np
import numpy.typing as npt

# Marks a wire dict as an encoded ndarray, alongside its dtype, shape and bytes.
_NDARRAY_MARKER = "__ndarray__"

# Marks a wire dict as an encoded tuple. msgpack has no tuple type, and the
# shared code type-checks some payloads (e.g. a lidar ``(frame, points)``) as
# tuples, so the tag carries the tuple-ness across the round trip.
_TUPLE_MARKER = "__tuple__"


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
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [_to_wire(item) for item in value]}
    if isinstance(value, list):
        return [_to_wire(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} for the SIL wire format")


def _from_wire(value: typing.Any) -> typing.Any:
    if isinstance(value, dict):
        if value.get(_NDARRAY_MARKER) is True:
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            # frombuffer is read-only and shares the msgpack buffer; copy so the
            # caller can keep the array after the payload is released.
            return array.reshape(value["shape"]).copy()
        if _TUPLE_MARKER in value:
            return tuple(_from_wire(item) for item in value[_TUPLE_MARKER])
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    return value


def encode(message: dict[str, typing.Any]) -> bytes:
    """Encode one message into its wire bytes.

    Args:
        message: A message built by :mod:`lead.evaluation.sil.contract`.

    Returns:
        The msgpack payload.
    """
    return typing.cast("bytes", msgpack.packb(_to_wire(message), use_bin_type=True))


def decode(payload: bytes) -> dict[str, typing.Any]:
    """Decode wire bytes back into a message.

    Args:
        payload: Bytes produced by :func:`encode`.

    Returns:
        The decoded message.
    """
    decoded = msgpack.unpackb(payload, raw=False)
    message: dict[str, typing.Any] = _from_wire(decoded)
    return message


def array_bytes(array: npt.NDArray) -> int:
    """Size of an array's payload in bytes, for bandwidth logging.

    Args:
        array: The array to measure.

    Returns:
        The number of bytes its data occupies.
    """
    return int(array.nbytes)
