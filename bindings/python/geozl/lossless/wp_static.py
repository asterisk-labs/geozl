import struct

from openzl import ext as _ext

from .._codec import _per_plane, planes_declared, row_width_declared
from .._ffi import _ptr, ffi, lib

_CTID = 0x72D707
_NAME = "geozl.lossless.wp_static"

# Little-endian width, shift, coefficients and optional plane count.
_HEADER = struct.Struct("<IB4h")
_HEADER_PLANES = struct.Struct("<IB4hI")

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric],
)


class _WpStaticEncoder(_ext.CustomEncoder):
    def __init__(self, width, planes=1):
        super().__init__()
        self._width = int(width)
        self._planes = int(planes)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        width = row_width_declared(self._width, n)
        planes = planes_declared(self._planes, width, n)
        src = _ptr(inp.content.as_nparray())

        coeffs = ffi.new("int16_t[4]")
        shift = ffi.new("uint8_t *")
        out = state.create_output(0, n, elt)
        # n == 0 is a valid empty stream, the kernels reject it as a geometry
        if n:
            # One set of weights is shared by all planes.
            if lib.wp_static_train(coeffs, shift, src, width, n, elt):
                raise ValueError(
                    f"{_NAME}: width {self._width} does not tile {n} samples")

            def enc(dst, s, w, nb, e):
                return lib.wp_static_encode(dst, s, w, nb, e, coeffs, shift[0])

            _per_plane(enc, _ptr(out.mut_content.as_nparray()), src, width, n,
                       elt, planes)
        state.send_codec_header(
            _HEADER_PLANES.pack(width, shift[0], coeffs[0], coeffs[1],
                                coeffs[2], coeffs[3], planes) if planes > 1
            else _HEADER.pack(width, shift[0], coeffs[0], coeffs[1],
                              coeffs[2], coeffs[3]))
        out.commit(n)


class WpStaticDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        inp = state.singleton_inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        raw = bytes(state.codec_header)
        if len(raw) == _HEADER.size:
            (width, shift, c0, c1, c2, c3), planes = _HEADER.unpack(raw), 1
        elif len(raw) == _HEADER_PLANES.size:
            width, shift, c0, c1, c2, c3, planes = _HEADER_PLANES.unpack(raw)
        else:
            raise ValueError(f"{_NAME}: bad codec header, {len(raw)} bytes "
                             f"where the frame carries {_HEADER.size} or "
                             f"{_HEADER_PLANES.size}")
        if shift >= 64:
            raise ValueError(f"{_NAME}: shift {shift} out of range")
        if planes == 0 or width == 0:
            raise ValueError(f"{_NAME}: codec header declares width {width} "
                             f"over {planes} planes")
        if planes > 1 and (n % planes or (n // planes) % width):
            raise ValueError(f"{_NAME}: {planes} planes do not split {n} "
                             f"samples into whole rows of {width}")
        coeffs = ffi.new("int16_t[]", [c0, c1, c2, c3])
        out = state.create_output(0, n, elt)

        def dec(dst, s, w, nb, e):
            return lib.wp_static_decode(dst, s, w, nb, e, coeffs, shift)

        if n and _per_plane(dec, _ptr(out.mut_content.as_nparray()),
                            _ptr(inp.content.as_nparray()), width, n, elt,
                            planes):
            raise ValueError(f"{_NAME}: bad row width in codec header")
        out.commit(n)


class WpStatic:
    def __init__(self, width, planes=1):
        self._width = int(width)
        self._planes = int(planes)

    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(
            _WpStaticEncoder(self._width, self._planes))
        return compressor.build_static_graph(node, [successor], name=_NAME)
