import struct

from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib

_CTID = 0x72D707
_NAME = "geozl.lossless.wp_static"

# Codec header, little endian, fixed layout: uint32 width, uint8 shift, then
# four int16 coefficients in the order cN, cNW, cNE, cNN. This is the only
# channel to the decoder, the C decode binding reads back exactly this.
_HEADER = "<IB4h"

_enc_kernel = lib.geozl_wp_static_encode
_dec_kernel = lib.geozl_wp_static_decode

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric],
)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, width, coeffs, shift):
        super().__init__()              # load bearing, inits the C++ side
        self._width = int(width)
        self._coeffs = tuple(int(c) for c in coeffs)
        self._shift = int(shift)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        state.send_codec_header(
            struct.pack(_HEADER, self._width, self._shift, *self._coeffs))
        coeffs = ffi.new("int16_t[4]", self._coeffs)
        out = state.create_output(0, n, elt)
        _enc_kernel(_ptr(out.mut_content.as_nparray()),
                    _ptr(inp.content.as_nparray()),
                    self._width, n, elt, coeffs, self._shift)
        out.commit(n)


class WpStaticDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        inp = state.singleton_inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        width, shift, *coeffs = struct.unpack(_HEADER, state.codec_header)
        c = ffi.new("int16_t[4]", coeffs)
        out = state.create_output(0, n, elt)
        _dec_kernel(_ptr(out.mut_content.as_nparray()),
                    _ptr(inp.content.as_nparray()),
                    width, n, elt, c, shift)
        out.commit(n)


class WpStaticInt:
    """Chains like an openzl.ext node. Frozen adaptation of the JPEG XL weighted
    predictor, W plus a linear kernel over the row above trained offline and
    carried in the codec header. coeffs is (cN, cNW, cNE, cNN). The defaults
    reproduce planar, so a trained kernel can only match it or beat it."""

    def __init__(self, width, coeffs=(1, -1, 0, 0), shift=0):
        self._width = int(width)
        self._coeffs = tuple(int(c) for c in coeffs)
        self._shift = int(shift)

    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(
            _Encoder(self._width, self._coeffs, self._shift))
        return compressor.build_static_graph(node, [successor], name=_NAME)