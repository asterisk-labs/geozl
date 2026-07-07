import struct

from openzl import ext as _ext

from ._ffi import _ptr, lib

# uint32 little endian row width, the whole header of a spatial predictor. The C
# binding writes the same four bytes, see the codec spec.
_HEADER = struct.Struct("<I")


def _row_width(width, n):
    """Effective row width, or 0 when width cannot tile n samples. Mirrors
    geozl_row_width on the C side. Call with n != 0."""
    w = n if (width == 0 or width > n) else width
    return w if n % w == 0 else 0


def spatial_predictor(ctid, name, encode, decode):
    """Node and decoder for a lossless spatial predictor, one numeric stream in
    and one out, parameterized by a row width. encode and decode are the C kernel
    names. Both take (dst, src, width, nb_elts, elt_width)."""
    enc = getattr(lib, encode)
    dec = getattr(lib, decode)
    short = name.rsplit(".", 1)[-1]

    desc = _ext.MultiInputCodecDescription(
        id=ctid,
        name=name,
        input_types=[_ext.Type.Numeric],
        singleton_output_types=[_ext.Type.Numeric],
    )

    class Encoder(_ext.CustomEncoder):
        def __init__(self, width):
            super().__init__()
            self._width = int(width)

        def multi_input_description(self):
            return desc

        def encode(self, state):
            inp = state.inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            if n and _row_width(self._width, n) == 0:
                raise ValueError(
                    f"{name}: width {self._width} does not tile {n} samples")
            out = state.create_output(0, n, elt)
            enc(_ptr(out.mut_content.as_nparray()),
                _ptr(inp.content.as_nparray()), self._width, n, elt)
            state.send_codec_header(_HEADER.pack(self._width))
            out.commit(n)

    class Decoder(_ext.CustomDecoder):
        def multi_input_description(self):
            return desc

        def decode(self, state):
            inp = state.singleton_inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            (width,) = _HEADER.unpack(state.codec_header)
            if n and _row_width(width, n) == 0:
                raise ValueError(f"{name}: bad row width in codec header")
            out = state.create_output(0, n, elt)
            dec(_ptr(out.mut_content.as_nparray()),
                _ptr(inp.content.as_nparray()), width, n, elt)
            out.commit(n)

    class Node:
        def __init__(self, width):
            self._width = int(width)

        def __call__(self, compressor, successor):
            if not isinstance(successor, _ext.GraphID):
                successor = successor.parameterize(compressor)
            node = compressor.register_custom_encoder(Encoder(self._width))
            return compressor.build_static_graph(node, [successor], name=name)

    Encoder.__name__ = Encoder.__qualname__ = f"_{short}_encoder"
    Decoder.__name__ = Decoder.__qualname__ = f"_{short}_decoder"
    Node.__name__ = Node.__qualname__ = f"_{short}_node"
    return Node, Decoder