import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, lib

_CTID = 0x72D704
_NAME = "geozl.lossless.deinterleave"

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)

_COMPONENT = {np.complex64: np.float32, np.complex128: np.float64}


def component_dtype(dtype):
    """The real component dtype a complex array views as, float32 for complex64.
    OpenZL has no complex type, so view a complex tile through this before the
    graph, and view the decoded output back to complex afterwards. deinterleave
    itself never sees the complex type, it only separates the two lanes."""
    return np.dtype(_COMPONENT[np.dtype(dtype).type])


class _Encoder(_ext.CustomEncoder):
    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        n0, n1 = (n + 1) // 2, n // 2
        out0 = state.create_output(0, n0, elt)
        out1 = state.create_output(1, n1, elt)
        lib.deinterleave_split(_ptr(out0.mut_content.as_nparray()),
                               _ptr(out1.mut_content.as_nparray()),
                               _ptr(inp.content.as_nparray()), n, elt)
        out0.commit(n0)
        out1.commit(n1)


class DeinterleaveDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        a = state.singleton_inputs[0]
        b = state.singleton_inputs[1]
        n = a.num_elts + b.num_elts
        out = state.create_output(0, n, a.elt_width)
        lib.deinterleave_join(_ptr(out.mut_content.as_nparray()),
                              _ptr(a.content.as_nparray()),
                              _ptr(b.content.as_nparray()), n, a.elt_width)
        out.commit(n)


class Deinterleave:
    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(_Encoder())
        return compressor.build_static_graph(
            node, [successor, successor], name=_NAME)
