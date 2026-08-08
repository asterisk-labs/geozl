import struct

from openzl import ext as _ext

from .._dtype import nodata_bits
from .._ffi import _ptr, ffi, lib

_CTID = 0x72D70C
_NAME = "geozl.lossless.nodata"

# What the kernels switch on. Anything else falls through their default and
# leaves the buffer untouched.
_WIDTHS = (1, 2, 4, 8)


_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)


def _bad_header(header, vals, mask):
    """Every check in the decoder means the same thing, that the header and the
    two stream sizes do not agree, so the sizes are the message."""
    return ValueError(
        f"{_NAME}: bad codec header, {len(header)} bytes, "
        f"{vals.num_elts} values, {mask.num_elts} mask")


def _pattern_bytes(pattern, elt):
    return struct.pack("<Q", pattern & 0xFFFFFFFFFFFFFFFF)[:elt]


class _Encoder(_ext.CustomEncoder):
    def __init__(self, width, pattern):
        super().__init__()
        self._width = int(width)
        self._pattern = None if pattern is None else int(pattern)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        if elt not in _WIDTHS:
            raise ValueError(f"{_NAME}: {elt}-byte elements, the kernels take "
                             f"1, 2, 4 or 8")
        # An empty tile has no mask to carry and the decoder refuses one, so
        # writing it would make a frame nothing can read.
        if n == 0:
            raise ValueError(f"{_NAME}: empty tile")
        vals = state.create_output(0, n, elt)
        mask = state.create_output(1, n, 1)
        src = _ptr(inp.content.as_nparray())
        mp = _ptr(mask.mut_content.as_nparray())
        if self._pattern is None:
            # No pattern given, so every NaN is a hole and the first one
            # supplies the pattern the decoder writes back.
            pattern = 0
            found = ffi.new("uint64_t*")
            if lib.nodata_find_nan(found, src, n, elt):
                pattern = int(found[0])
            lib.nodata_mark_nan(mp, src, n, elt)
        else:
            pattern = self._pattern
            lib.nodata_mark_value(mp, src, n, elt, pattern)

        state.send_codec_header(_pattern_bytes(pattern, elt))
        lib.nodata_fill(_ptr(vals.mut_content.as_nparray()), src, mp,
                        self._width, n, elt)
        vals.commit(n)
        mask.commit(n)


class NodataDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        vals = state.singleton_inputs[0]
        mask = state.singleton_inputs[1]
        elt = vals.elt_width
        if elt not in _WIDTHS or mask.elt_width != 1:
            raise ValueError(f"{_NAME}: bad stream widths, {elt} values, "
                             f"{mask.elt_width} mask")
        header = state.codec_header
        n = vals.num_elts
        if len(header) != elt or mask.num_elts != n or n == 0:
            raise _bad_header(header, vals, mask)
        pattern = int.from_bytes(header, "little")
        out = state.create_output(0, n, elt)
        lib.nodata_restore(_ptr(out.mut_content.as_nparray()),
                           _ptr(vals.content.as_nparray()),
                           _ptr(mask.content.as_nparray()), n, elt, pattern)
        out.commit(n)


class Nodata:
    """Pulls the samples that were never measured out into a validity mask and
    fills the holes, so whatever runs next sees a raster with no cliff at the
    edge of a hole.

    A tile with nothing missing still pays for a mask, so put the node in the
    graph for the tiles that need it and leave it out of the ones that do not.
    """

    def __init__(self, width, value=None, dtype=None):
        self._width = int(width)
        if value is None:
            self._pattern = None
        elif dtype is None:
            raise ValueError("give dtype alongside value, a sentinel's bit "
                             "pattern depends on the type it is read at")
        else:
            self._pattern = nodata_bits(value, dtype)

    def __call__(self, compressor, values_successor, mask_successor):
        succ = []
        for s in (values_successor, mask_successor):
            succ.append(s if isinstance(s, _ext.GraphID)
                        else s.parameterize(compressor))
        node = compressor.register_custom_encoder(
            _Encoder(self._width, self._pattern))
        return compressor.build_static_graph(node, succ, name=_NAME)