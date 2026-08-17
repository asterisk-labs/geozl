import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib

_CTID = 0x72D70D
_NAME = "geozl.lossless.pfor"

_WIDTHS = (1, 2, 4, 8)

_HEADER = struct.Struct("<QB")

# Maximum elements described per byte by a minimum-size block.
_MAX_ELTS_PER_BYTE = 256 // 2

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Serial],
)


class _Encoder(_ext.CustomEncoder):
    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        if elt not in _WIDTHS:
            raise ValueError(f"{_NAME}: {elt}-byte elements, the kernels take "
                             f"1, 2, 4 or 8")
        state.send_codec_header(_HEADER.pack(n, elt))

        if n == 0:
            out = state.create_output(0, 1, 1)
            out.commit(0)
            return

        bound = lib.pfor_bound(n, elt)
        if bound == 0:
            raise ValueError(f"{_NAME}: geometry refused, {n} x {elt}")
        out = state.create_output(0, bound, 1)
        written = ffi.new("size_t*")
        if lib.pfor_encode(_ptr(out.mut_content.as_nparray()), bound, written,
                           _ptr(inp.content.as_nparray()), n, elt):
            raise ValueError(f"{_NAME}: encode refused, {n} x {elt}")
        out.commit(int(written[0]))


class PforDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        src = state.singleton_inputs[0]
        header = state.codec_header
        if len(header) != _HEADER.size:
            raise ValueError(f"{_NAME}: bad codec header, {len(header)} bytes")
        n, elt = _HEADER.unpack(bytes(header))
        if elt not in _WIDTHS:
            raise ValueError(f"{_NAME}: bad element width {elt}")

        size = src.num_elts
        if n == 0:
            if size != 0:
                raise ValueError(f"{_NAME}: empty tile with {size} bytes")
            out = state.create_output(0, 0, elt)
            out.commit(0)
            return

        if size == 0 or n > size * _MAX_ELTS_PER_BYTE:
            raise ValueError(f"{_NAME}: {n} elements do not fit {size} bytes")

        out = state.create_output(0, n, elt)
        if lib.pfor_decode(_ptr(out.mut_content.as_nparray()), n, elt,
                           _ptr(src.content.as_nparray()), size):
            raise ValueError(f"{_NAME}: corrupt stream")
        out.commit(n)


class Pfor:
    """Pack fixed-size blocks and patch values wider than their bit width."""

    def __call__(self, compressor, successor):
        succ = (successor if isinstance(successor, _ext.GraphID)
                else successor.parameterize(compressor))
        node = compressor.register_custom_encoder(_Encoder())
        return compressor.build_static_graph(node, [succ], name=_NAME)


def bound(nb_elts, elt_width):
    """Worst-case output size, or 0 for invalid geometry."""
    return int(lib.pfor_bound(int(nb_elts), int(elt_width)))


def encode(values):
    """Pack a 1-D numeric array. Returns the packed bytes."""
    a = np.ascontiguousarray(values)
    n, elt = a.size, a.dtype.itemsize
    cap = bound(n, elt)
    if cap == 0:
        raise ValueError(f"{_NAME}: geometry refused, {n} x {elt}")
    dst = np.empty(cap, dtype=np.uint8)
    written = ffi.new("size_t*")
    if lib.pfor_encode(_ptr(dst), cap, written, _ptr(a), n, elt):
        raise ValueError(f"{_NAME}: encode refused")
    return dst[:int(written[0])].tobytes()


def decode(packed, nb_elts, dtype):
    """Unpack into a 1-D array of nb_elts values of dtype."""
    dt = np.dtype(dtype)
    out = np.empty(nb_elts, dtype=dt)
    buf = np.frombuffer(packed, dtype=np.uint8)
    if lib.pfor_decode(_ptr(out), nb_elts, dt.itemsize, _ptr(buf), buf.size):
        raise ValueError(f"{_NAME}: corrupt stream")
    return out


__all__ = ["Pfor", "PforDecoder", "bound", "encode", "decode"]
