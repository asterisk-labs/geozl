from openzl import ext as _ext

from .._ffi import _ptr, lib


def predictor(ctid, name, enc_name, dec_name):
    enc_kernel = getattr(lib, enc_name)
    dec_kernel = getattr(lib, dec_name)
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
            state.send_codec_header(self._width.to_bytes(4, "little"))
            out = state.create_output(0, n, elt)
            enc_kernel(_ptr(out.mut_content.as_nparray()),
                       _ptr(inp.content.as_nparray()),
                       self._width, n, elt)
            out.commit(n)

    class Decoder(_ext.CustomDecoder):
        def multi_input_description(self):
            return desc

        def decode(self, state):
            inp = state.singleton_inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            width = int.from_bytes(state.codec_header, "little")
            out = state.create_output(0, n, elt)
            dec_kernel(_ptr(out.mut_content.as_nparray()),
                       _ptr(inp.content.as_nparray()),
                       width, n, elt)
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
