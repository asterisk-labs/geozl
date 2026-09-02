import struct

import numpy as np
from openzl import ext as _ext

from .._codec import planes_declared, row_width_declared
from .._ffi import _ptr, ffi, lib

_CTID = 0x72D710
_NAME = "geozl.lossless.planar_zigzag_pfor"
_WIDTHS = (1, 2, 4, 8)
_HEADER = struct.Struct("<QBII")
_MAX_ELTS_PER_BYTE = 256 // 2

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Serial],
)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, width, planes=1):
        super().__init__()
        self._width = int(width)
        self._planes = int(planes)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        if elt not in _WIDTHS:
            raise ValueError(f"{_NAME}: bad element width {elt}")
        width = row_width_declared(self._width, n)
        planes = planes_declared(self._planes, width, n)
        state.send_codec_header(_HEADER.pack(n, elt, width, planes))

        if n == 0:
            out = state.create_output(0, 1, 1)
            out.commit(0)
            return

        bound = lib.planar_zigzag_pfor_bound(n, elt)
        if bound == 0:
            raise ValueError(f"{_NAME}: geometry refused, {n} x {elt}")
        out = state.create_output(0, bound, 1)
        written = ffi.new("size_t*")
        if lib.planar_zigzag_pfor_encode(
            _ptr(out.mut_content.as_nparray()),
            bound,
            written,
            _ptr(inp.content.as_nparray()),
            width,
            n,
            elt,
            planes,
        ):
            raise ValueError(
                f"{_NAME}: width {width} over {planes} planes refused"
            )
        out.commit(int(written[0]))


class PlanarZigzagPforDecoder(_ext.CustomDecoder):
    """OpenZL decoder registration for :class:`PlanarZigzagPfor` frames."""

    def multi_input_description(self):
        return _desc

    def decode(self, state):
        src = state.singleton_inputs[0]
        raw = bytes(state.codec_header)
        if len(raw) != _HEADER.size:
            raise ValueError(f"{_NAME}: bad codec header, {len(raw)} bytes")
        n, elt, width, planes = _HEADER.unpack(raw)
        if elt not in _WIDTHS or planes == 0:
            raise ValueError(f"{_NAME}: bad element width or plane count")

        size = src.num_elts
        if n == 0:
            if size != 0:
                raise ValueError(f"{_NAME}: empty tile with {size} bytes")
            out = state.create_output(0, 0, elt)
            out.commit(0)
            return
        if (
            width == 0
            or n % planes
            or width > n // planes
            or (n // planes) % width
        ):
            raise ValueError(f"{_NAME}: bad row width or plane layout")
        if size == 0 or n > size * _MAX_ELTS_PER_BYTE:
            raise ValueError(f"{_NAME}: {n} elements do not fit {size} bytes")

        out = state.create_output(0, n, elt)
        if lib.planar_zigzag_pfor_decode(
            _ptr(out.mut_content.as_nparray()),
            width,
            n,
            elt,
            planes,
            _ptr(src.content.as_nparray()),
            size,
        ):
            raise ValueError(f"{_NAME}: corrupt stream or geometry")
        out.commit(n)


class PlanarZigzagPfor:
    """Build the fused Planar, Zigzag and PFOR codec.

    The node emits a serial stream and must therefore lead to a serial sink.
    Its raw payload matches ``PlanarZigzag`` followed by ``Pfor``, while its
    CTID and codec header are distinct. Decoding uses a fixed 256-value scratch
    block instead of materialising full transformed arrays.

    Args:
        width: Row width in samples.
        planes: Number of contiguous, independently predicted planes.
    """

    def __init__(self, width, planes=1):
        self._width = int(width)
        self._planes = int(planes)

    def __call__(self, compressor, successor):
        succ = (successor if isinstance(successor, _ext.GraphID)
                else successor.parameterize(compressor))
        node = compressor.register_custom_encoder(_Encoder(self._width,
                                                            self._planes))
        return compressor.build_static_graph(node, [succ], name=_NAME)


def bound(nb_elts, elt_width):
    """Return the worst-case payload size, or zero for invalid/empty input."""
    return int(lib.planar_zigzag_pfor_bound(int(nb_elts), int(elt_width)))


def encode(values, width, planes=1):
    """Encode values to the raw payload, without an OpenZL frame header."""
    flat = np.ascontiguousarray(values).reshape(-1)
    n, elt = flat.size, flat.dtype.itemsize
    row_width = row_width_declared(int(width), n)
    plane_count = planes_declared(int(planes), row_width, n)
    cap = bound(n, elt)
    if cap == 0:
        raise ValueError(f"{_NAME}: geometry refused, {n} x {elt}")
    dst = np.empty(cap, dtype=np.uint8)
    written = ffi.new("size_t*")
    if lib.planar_zigzag_pfor_encode(
        _ptr(dst), cap, written, _ptr(flat), row_width, n, elt, plane_count
    ):
        raise ValueError(f"{_NAME}: geometry refused")
    return dst[: int(written[0])].tobytes()


def decode(packed, nb_elts, dtype, width, planes=1):
    """Decode a raw payload into a flat array of ``dtype``."""
    dt = np.dtype(dtype)
    out = np.empty(nb_elts, dtype=dt)
    buf = np.frombuffer(packed, dtype=np.uint8)
    if lib.planar_zigzag_pfor_decode(
        _ptr(out), width, nb_elts, dt.itemsize, planes, _ptr(buf), buf.size
    ):
        raise ValueError(f"{_NAME}: corrupt stream or geometry")
    return out


__all__ = ["PlanarZigzagPfor", "PlanarZigzagPforDecoder", "bound", "encode", "decode"]
