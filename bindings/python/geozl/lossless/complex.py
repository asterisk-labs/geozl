import numpy as np
from openzl import ext as _ext


_COMPLEX_CTID = 0x72D704
_NAME = "geozl.lossless.complex"

# code -> numpy complex dtype, rides in the codec header so decode rebuilds the
# right one. Order is the wire value.
_DTYPE = {0: np.complex64, 1: np.complex128}
_CODE = {np.dtype(v): k for k, v in _DTYPE.items()}
_COMPONENT = {np.complex64: np.float32, np.complex128: np.float64}


def component_dtype(dtype):
    """Real component dtype a complex views as, float32 for complex64. OpenZL
    has no complex type, so view the input through this before handing it over."""
    return np.dtype(_COMPONENT[np.dtype(dtype).type])


_desc = _ext.MultiInputCodecDescription(
    id=_COMPLEX_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, code):
        super().__init__()              # load bearing, inits the C++ side
        self._code = code

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        arr = inp.content.as_nparray()          # components, [re, im, re, im]
        half = inp.num_elts // 2
        state.send_codec_header(bytes([self._code]))
        re = state.create_output(0, half, inp.elt_width)
        im = state.create_output(1, half, inp.elt_width)
        re.mut_content.as_nparray()[:] = arr[0::2]
        im.mut_content.as_nparray()[:] = arr[1::2]
        re.commit(half)
        im.commit(half)


class ComplexSplitDecoder(_ext.CustomDecoder):
    def __init__(self):
        super().__init__()
        self.dtype = None               # the caller reads this to view the output

    def multi_input_description(self):
        return _desc

    def decode(self, state):
        re = state.singleton_inputs[0].content.as_nparray()
        im = state.singleton_inputs[1].content.as_nparray()
        elt = state.singleton_inputs[0].elt_width
        self.dtype = _DTYPE[state.codec_header[0]]
        n = re.shape[0]
        out = state.create_output(0, 2 * n, elt)
        o = out.mut_content.as_nparray()
        o[0::2] = re
        o[1::2] = im
        out.commit(2 * n)


def complex_split(dtype):
    """Head of graph lossless node for complex data. Splits real and imaginary
    into two streams so a spatial predictor never crosses the component boundary.
    Both streams chain to the same successor."""
    code = _CODE.get(np.dtype(dtype))
    if code is None:
        raise ValueError(f"complex_split does not support dtype {dtype!r}")

    def build(compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(_Encoder(code))
        return compressor.build_static_graph(
            node, [successor, successor], name=_NAME)

    return build


def decompress_complex(frame, dtype=None):
    """Decompress a complex_split frame back to its complex array. OpenZL has no
    complex type, so it decodes to component floats and this views them back.
    Pass dtype when the frame stored the data uncompressed, which drops the codec
    that would otherwise carry the complex type."""
    from . import _DECODERS

    dctx = _ext.DCtx()
    decoder = ComplexSplitDecoder()
    dctx.register_custom_decoder(decoder)
    for other in _DECODERS:
        if other is not ComplexSplitDecoder:
            dctx.register_custom_decoder(other())

    out = dctx.decompress(frame)
    recovered = decoder.dtype or dtype
    if recovered is None:
        raise ValueError(
            "complex dtype not in the frame (stored uncompressed), "
            "pass dtype to recover it")
    return out[0].content.as_nparray().view(recovered)