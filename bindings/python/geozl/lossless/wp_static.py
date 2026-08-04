import struct

from openzl import ext as _ext

from .._codec import row_width_declared
from .._ffi import _ptr, ffi, lib

_CTID = 0x72D707
_NAME = "geozl.lossless.wp_static"

# uint32 width, uint8 shift, four int16 {cN, cNW, cNE, cNN}, little endian.
_HEADER = struct.Struct("<IB4h")

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric],
)


class _WpStaticEncoder(_ext.CustomEncoder):
    def __init__(self, width):
        super().__init__()
        self._width = int(width)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        width = row_width_declared(self._width, n)
        src = _ptr(inp.content.as_nparray())

        coeffs = ffi.new("int16_t[4]")
        shift = ffi.new("uint8_t *")
        out = state.create_output(0, n, elt)
        # n == 0 is a valid empty stream, the kernels reject it as a geometry
        if n:
            if lib.wp_static_train(coeffs, shift, src, width, n, elt):
                raise ValueError(
                    f"{_NAME}: width {self._width} does not tile {n} samples")
            lib.wp_static_encode(_ptr(out.mut_content.as_nparray()), src,
                                 width, n, elt, coeffs, shift[0])
        state.send_codec_header(
            _HEADER.pack(width, shift[0], coeffs[0], coeffs[1],
                         coeffs[2], coeffs[3]))
        out.commit(n)


class WpStaticDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        inp = state.singleton_inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        width, shift, c0, c1, c2, c3 = _HEADER.unpack(state.codec_header)
        if shift >= 64:
            raise ValueError(f"{_NAME}: shift {shift} out of range")
        coeffs = ffi.new("int16_t[]", [c0, c1, c2, c3])
        out = state.create_output(0, n, elt)
        if n and lib.wp_static_decode(_ptr(out.mut_content.as_nparray()),
                                      _ptr(inp.content.as_nparray()), width, n,
                                      elt, coeffs, shift):
            raise ValueError(f"{_NAME}: bad row width in codec header")
        out.commit(n)


class WpStatic:
    def __init__(self, width):
        self._width = int(width)

    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(_WpStaticEncoder(self._width))
        return compressor.build_static_graph(node, [successor], name=_NAME)