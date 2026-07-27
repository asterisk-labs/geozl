import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib

_CTID = 0x72D70C
_NAME = "geozl.lossless.nodata"

# uint8 code, then the bit pattern at the sample width. The C binding writes the
# same bytes, see the codec spec.
_RESTORE = 1

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)

INVALID, VALID = 0, 255


def _pattern_bytes(pattern, elt):
    return struct.pack("<Q", pattern & 0xFFFFFFFFFFFFFFFF)[:elt]


def nodata_bits(value, dtype):
    """Bit pattern a nodata value has at its own dtype, which is what the codec
    header carries. A float keeps its exact bits, so a NaN payload survives."""
    return int(np.asarray(value, dtype=np.dtype(dtype)).view(
        np.dtype(f"u{np.dtype(dtype).itemsize}")))


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
        vals = state.create_output(0, n, elt)
        mask = state.create_output(1, n, 1)
        pattern = self._pattern or 0
        if n:
            src = _ptr(inp.content.as_nparray())
            mp = _ptr(mask.mut_content.as_nparray())
            if self._pattern is None:
                # No pattern given, so every NaN is a hole and the first one
                # supplies the pattern the decoder writes back.
                found = ffi.new("uint64_t*")
                if lib.nodata_find_nan(found, src, n, elt):
                    pattern = int(found[0])
                lib.nodata_mark_nan(mp, src, n, elt)
            else:
                lib.nodata_mark_value(mp, src, n, elt, pattern)
            lib.nodata_fill(_ptr(vals.mut_content.as_nparray()), src, mp,
                            self._width, n, elt)
        state.send_codec_header(bytes([_RESTORE]) + _pattern_bytes(pattern, elt))
        vals.commit(n)
        mask.commit(n)


class NodataDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        vals = state.singleton_inputs[0]
        mask = state.singleton_inputs[1]
        n, elt = vals.num_elts, vals.elt_width
        header = state.codec_header
        if len(header) != 1 + elt or header[0] != _RESTORE:
            raise ValueError(f"{_NAME}: bad codec header")
        pattern = int.from_bytes(header[1:], "little")
        out = state.create_output(0, n, elt)
        lib.nodata_restore(_ptr(out.mut_content.as_nparray()),
                           _ptr(vals.content.as_nparray()),
                           _ptr(mask.content.as_nparray()), n, elt, pattern)
        out.commit(n)


class Nodata:
    """Pulls the samples that were never measured out into a validity mask and
    fills the holes, so whatever runs next sees a raster with no cliff at the
    edge of a hole.
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