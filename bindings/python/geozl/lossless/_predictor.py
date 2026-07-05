from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib


def predictor(ctid, name, encode_auto, decode_auto, header_cap):
    """Build the Node and Decoder for a lossless predictor whose header is owned
    by C. encode_auto estimates any parameters, writes the header, and encodes.
    decode_auto reads the header and decodes. Python only moves the opaque header
    bytes, it never reads or writes the layout."""
    enc = getattr(lib, encode_auto)
    dec = getattr(lib, decode_auto)
    short = name.rsplit(".", 1)[-1]

    desc = _ext.MultiInputCodecDescription(
        id=ctid,
        name=name,
        input_types=[_ext.Type.Numeric],
        singleton_output_types=[_ext.Type.Numeric],
    )

    class Encoder(_ext.CustomEncoder):
        def __init__(self, width):
            super().__init__()          # load bearing, inits the C++ side
            self._width = int(width)

        def multi_input_description(self):
            return desc

        def encode(self, state):
            inp = state.inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            out = state.create_output(0, n, elt)
            header = ffi.new("uint8_t[]", header_cap)
            size = enc(_ptr(out.mut_content.as_nparray()), header, header_cap,
                       _ptr(inp.content.as_nparray()), self._width, n, elt)
            state.send_codec_header(bytes(ffi.buffer(header, size)))
            out.commit(n)

    class Decoder(_ext.CustomDecoder):
        def multi_input_description(self):
            return desc

        def decode(self, state):
            inp = state.singleton_inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            head = state.codec_header
            header = ffi.new("uint8_t[]", head)
            out = state.create_output(0, n, elt)
            dec(_ptr(out.mut_content.as_nparray()), _ptr(inp.content.as_nparray()),
                header, len(head), n, elt)
            out.commit(n)

    class Node:
        """Chains like an openzl.ext node: call with the compressor and the
        successor graph, get back a GraphID."""

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
