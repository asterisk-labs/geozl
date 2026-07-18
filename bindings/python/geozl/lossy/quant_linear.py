import math
import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, lib
from ._base import require_checksum_disabled

_CTID = 0x72D780
_NAME = "geozl.lossy.quant_linear"

# Mirrors the ql_dtype enum in graph_quant_linear.h, the value is the wire code.
_QL_DTYPE = {
    np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2,
    np.dtype("uint64"): 3, np.dtype("int8"): 4, np.dtype("int16"): 5,
    np.dtype("int32"): 6, np.dtype("int64"): 7, np.dtype("float16"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10,
}

# Element width per dtype code, derived from the table above so it cannot drift
# from it. Codes are 0..N-1, so a tuple indexes straight by code.
_WIDTH = tuple(dt.itemsize
               for dt, _c in sorted(_QL_DTYPE.items(), key=lambda kv: kv[1]))


def dtype_code(dtype):
    """The ql_dtype wire code for dtype, or None when quant_linear has no kernel
    for it. Keyed on the numpy dtype, so a byte-swapped array is refused rather
    than quantized as native."""
    return _QL_DTYPE.get(np.dtype(dtype))

# uint8 dtype code, then the scale as an IEEE double, little endian.
_HEADER = struct.Struct("<Bd")

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric],
)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, scale, dtype):
        super().__init__()
        self._scale = scale
        self._dtype = dtype

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        require_checksum_disabled(state, _NAME)
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        if _WIDTH[self._dtype] != elt:
            raise ValueError(
                f"{_NAME}: dtype {self._dtype} does not match {elt}-byte samples")
        out = state.create_output(0, n, elt)
        lib.quant_linear_encode(_ptr(out.mut_content.as_nparray()),
                                _ptr(inp.content.as_nparray()),
                                self._scale, self._dtype, n)
        state.send_codec_header(_HEADER.pack(self._dtype, self._scale))
        out.commit(n)


class QuantLinearDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        inp = state.singleton_inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        dtype, scale = _HEADER.unpack(bytes(state.codec_header))
        if not 0 <= dtype < len(_WIDTH) or _WIDTH[dtype] != elt:
            raise ValueError(f"{_NAME}: bad dtype in codec header")
        if not math.isfinite(scale):
            raise ValueError(f"{_NAME}: bad scale in codec header")
        out = state.create_output(0, n, elt)
        lib.quant_linear_decode(_ptr(out.mut_content.as_nparray()),
                                _ptr(inp.content.as_nparray()),
                                scale, dtype, n)
        out.commit(n)


class QuantLinear:
    """Head of graph lossy node. Quantizes to a uniform step of 2*max_error,
    which bounds the per element error by max_error. Disable ContentChecksum on
    the CCtx, the round trip is not bit exact."""

    def __init__(self, max_error, dtype):
        code = dtype_code(dtype)
        if code is None:
            raise ValueError(f"quant_linear does not support dtype {dtype!r}")
        scale = 2.0 * float(max_error)
        if scale <= 0.0:
            raise ValueError("max_error must be positive")
        self._scale = scale
        self._dtype = code

    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(
            _Encoder(self._scale, self._dtype))
        return compressor.build_static_graph(node, [successor], name=_NAME)