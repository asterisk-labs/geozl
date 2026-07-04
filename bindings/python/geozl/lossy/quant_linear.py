import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, lib
from ._base import require_checksum_disabled


_CTID = 0x72D780
_NAME = "geozl.lossy.quant_linear"

# Mirrors the ql_dtype enum in graph_quant_linear.h. Order is the wire value.
_QL_DTYPE = {
    np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2,
    np.dtype("uint64"): 3, np.dtype("int8"): 4, np.dtype("int16"): 5,
    np.dtype("int32"): 6, np.dtype("int64"): 7, np.dtype("float16"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10,
}

_HEADER = struct.Struct("<Bd")          # dtype byte, scale as f64


_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric],
)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, scale, dtype):
        super().__init__()              # load bearing, inits the C++ side
        self._scale = scale
        self._dtype = dtype

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        require_checksum_disabled(state, _NAME)
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        state.send_codec_header(_HEADER.pack(self._dtype, self._scale))
        out = state.create_output(0, n, elt)
        lib.geozl_quant_linear_encode(
            _ptr(out.mut_content.as_nparray()),
            _ptr(inp.content.as_nparray()),
            self._scale, self._dtype, n)
        out.commit(n)


class QuantLinearDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        inp = state.singleton_inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        dtype, scale = _HEADER.unpack(bytes(state.codec_header))
        out = state.create_output(0, n, elt)
        lib.geozl_quant_linear_decode(
            _ptr(out.mut_content.as_nparray()),
            _ptr(inp.content.as_nparray()),
            scale, dtype, n)
        out.commit(n)


def quant_linear(max_error, dtype):
    """Head of graph lossy node. Quantizes to a uniform step of 2*max_error,
    bounding the per pixel error by max_error. Omit it for lossless. Disable
    ContentChecksum on the CCtx, the round trip is not bit exact."""
    qd = _QL_DTYPE.get(np.dtype(dtype))
    if qd is None:
        raise ValueError(f"quant_linear does not support dtype {dtype!r}")
    scale = 2.0 * float(max_error)
    if scale <= 0.0:
        raise ValueError("max_error must be positive")

    def build(compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(_Encoder(scale, qd))
        return compressor.build_static_graph(node, [successor], name=_NAME)

    return build
