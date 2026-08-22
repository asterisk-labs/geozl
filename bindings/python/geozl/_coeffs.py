import operator
from collections.abc import Iterable

import numpy as np

from ._ffi import _load_lib_full, _ptr, ffi
from ._ffi import lib as _kernels

CoeffVectors = Iterable[Iterable[int]]

_MAX_BYTES = 10_000
_MAX_VECTORS = 255
_NOT_A_VECTOR = (str, bytes, bytearray, memoryview)
_INT32_MIN = -(2 ** 31)
_INT32_MAX = 2 ** 31 - 1


def _pack_coeffs(vectors: CoeffVectors) -> bytes:
    if isinstance(vectors, _NOT_A_VECTOR):
        raise TypeError("coeffs must be an iterable of numeric iterables")

    try:
        iterator = iter(vectors)
    except TypeError as exc:
        raise TypeError("coeffs must be an iterable of numeric iterables") from exc

    packed: list[list[int]] = []
    used = 6
    for vector in iterator:
        if len(packed) == _MAX_VECTORS:
            raise ValueError(f"coeffs supports at most {_MAX_VECTORS} vectors")
        if isinstance(vector, _NOT_A_VECTOR):
            raise TypeError("each coeffs vector must be an iterable of numbers")
        try:
            values = iter(vector)
        except TypeError as exc:
            raise TypeError("each coeffs vector must be an iterable of numbers") from exc

        row = []
        used += 4
        for value in values:
            if used + 4 > _MAX_BYTES:
                raise ValueError(f"coeffs exceeds the {_MAX_BYTES}-byte limit")
            if isinstance(value, bool):
                raise TypeError("coefficient must not be a boolean")
            # index() takes ints and numpy integers and rejects floats, rather
            # than truncating them into a different coefficient.
            try:
                number = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"coefficient is not an integer: {value!r}") from exc
            if not _INT32_MIN <= number <= _INT32_MAX:
                raise ValueError(f"coefficient does not fit in an int32: {number}")
            row.append(number)
            used += 4
        if not row:
            raise ValueError(f"coeffs vector {len(packed)} is empty")
        packed.append(row)

    if not packed:
        raise ValueError("coeffs must contain at least one vector")

    counts = ffi.new("uint32_t[]", [len(vector) for vector in packed])
    size = _kernels.geozl_coeffs_size(counts, len(packed))
    if size == 0:
        raise ValueError("coeffs has an invalid shape")

    flat = ffi.new("int32_t[]", max(sum(map(len, packed)), 1))
    pointers = ffi.new("int32_t*[]", len(packed))
    offset = 0
    for i, vector in enumerate(packed):
        pointers[i] = flat + offset
        for value in vector:
            flat[offset] = value
            offset += 1

    dst = ffi.new("char[]", size)
    err = ffi.new("char[]", 128)
    written = _kernels.geozl_coeffs_pack(
        dst, len(dst), pointers, counts, len(packed), err, len(err)
    )
    if written == 0:
        reason = ffi.string(err).decode("utf-8", "replace")
        raise ValueError(reason or "coeffs could not be packed")
    return bytes(ffi.buffer(dst, written))


def _parse_coeffs(blob: bytes) -> tuple[tuple[int, ...], ...] | None:
    vector_count = ffi.new("size_t*")
    value_count = ffi.new("size_t*")
    rc = _kernels.geozl_coeffs_parse(
        blob, len(blob), ffi.NULL, 0, ffi.NULL, 0, vector_count, value_count
    )
    if rc == 1:
        return None
    if rc != 0:
        raise ValueError(f"invalid geozl coefficient blob (code {rc})")

    counts = ffi.new("uint32_t[]", vector_count[0])
    values = ffi.new("int32_t[]", max(value_count[0], 1))
    rc = _kernels.geozl_coeffs_parse(
        blob,
        len(blob),
        values,
        value_count[0],
        counts,
        vector_count[0],
        ffi.NULL,
        ffi.NULL,
    )
    if rc != 0:
        raise ValueError(f"invalid geozl coefficient blob (code {rc})")

    result = []
    offset = 0
    for count in counts:
        result.append(tuple(values[offset:offset + count]))
        offset += count
    return tuple(result)


def coeffs(frame: bytes) -> tuple[tuple[int, ...], ...] | None:
    """Return coefficient vectors from ``frame``, or ``None`` if absent.

    Only the frame header is read; the payload is not decompressed.
    """
    full = _load_lib_full()
    src = np.frombuffer(frame, np.uint8)
    size = ffi.new("size_t*")
    # Query the size before allocating the Python-side buffer.
    rc = full.geozl_2d_frame_coeffs_c(_ptr(src), src.size, ffi.NULL, 0, size)
    if rc < 0:
        return None
    if rc != 0:
        raise RuntimeError(f"geozl.coeffs: unreadable frame (ZL error code {rc})")

    blob = ffi.new("char[]", size[0])
    rc = full.geozl_2d_frame_coeffs_c(
        _ptr(src), src.size, blob, size[0], size
    )
    if rc != 0:
        raise RuntimeError(f"geozl.coeffs: unreadable frame (ZL error code {rc})")
    return _parse_coeffs(bytes(ffi.buffer(blob, size[0])))
