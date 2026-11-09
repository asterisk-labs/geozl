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
    """Return the real component dtype for a complex dtype."""
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
    """Decoder for ``geozl.lossless.deinterleave`` frames."""

    def multi_input_description(self):
        return _desc

    def decode(self, state):
        a = state.singleton_inputs[0]
        b = state.singleton_inputs[1]
        if a.elt_width != b.elt_width or a.num_elts - b.num_elts not in (0, 1):
            raise ValueError(f"{_NAME}: lanes differ in width or length")
        n = a.num_elts + b.num_elts
        out = state.create_output(0, n, a.elt_width)
        lib.deinterleave_join(_ptr(out.mut_content.as_nparray()),
                              _ptr(a.content.as_nparray()),
                              _ptr(b.content.as_nparray()), n, a.elt_width)
        out.commit(n)


class Deinterleave:
    """Split a stream into its even and odd samples.

    Complex samples stored as real and imaginary pairs become one lane of
    each. Both lanes go to the one successor the node is called with.
    """

    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(_Encoder())
        return compressor.build_static_graph(
            node, [successor, successor], name=_NAME)
