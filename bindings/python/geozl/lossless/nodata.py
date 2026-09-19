import struct

from openzl import ext as _ext

from .._dtype import dtype_code, dtype_width, nodata_bits
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
    return ValueError(
        f"{_NAME}: bad codec header, {len(header)} bytes, "
        f"{vals.num_elts} values, {mask.num_elts} mask")


def _pattern_bytes(pattern, elt):
    return struct.pack("<Q", pattern & 0xFFFFFFFFFFFFFFFF)[:elt]


class _Encoder(_ext.CustomEncoder):
    def __init__(self, width, pattern, dtype=None, radius=None):
        super().__init__()
        self._width = int(width)
        self._pattern = None if pattern is None else int(pattern)
        self._dtype = dtype
        self._radius = float("inf") if radius is None else float(radius)

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
        if self._dtype is not None and dtype_width(self._dtype) != elt:
            raise ValueError(f"{_NAME}: dtype code {self._dtype} over "
                             f"{elt}-byte elements")
        vals = state.create_output(0, n, elt)
        mask = state.create_output(1, n, 1)
        src = _ptr(inp.content.as_nparray())
        mp = _ptr(mask.mut_content.as_nparray())
        guarded = False
        if self._pattern is None:
            # No pattern given, so every NaN is a hole and the first one
            # supplies the pattern the decoder writes back.
            pattern = 0
            found = ffi.new("uint64_t*")
            if lib.nodata_find_nan(found, src, n, elt):
                pattern = int(found[0])
            lib.nodata_mark_nan(mp, src, n, elt)
        else:
            pattern = self._pattern & ((1 << (8 * elt)) - 1)
            if self._dtype is None:
                lib.nodata_mark_value(mp, src, n, elt, pattern)
            else:
                rc = lib.nodata_mark_guarded(mp, src, n, self._dtype, pattern,
                                             self._radius)
                # NaN sentinels use the plain form.
                if rc not in (0, 2):
                    raise ValueError(f"{_NAME}: dtype code {self._dtype} is "
                                     f"not one geozl knows")
                guarded = rc == 0

        header = _pattern_bytes(pattern, elt)
        if guarded:
            repl = ffi.new("uint64_t[3]")
            lib.nodata_guard_values(repl, self._dtype, pattern)
            header += b"".join(_pattern_bytes(int(r), elt) for r in repl)
        state.send_codec_header(header)
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
        # The header length identifies the wire form.
        guarded = len(header) == 4 * elt
        if (len(header) != elt and not guarded) or mask.num_elts != n or n == 0:
            raise _bad_header(header, vals, mask)
        pattern = int.from_bytes(header[:elt], "little")
        out = state.create_output(0, n, elt)
        args = (_ptr(out.mut_content.as_nparray()),
                _ptr(vals.content.as_nparray()),
                _ptr(mask.content.as_nparray()), n, elt, pattern)
        if guarded:
            repl = ffi.new("uint64_t[3]", [
                int.from_bytes(header[k * elt:(k + 1) * elt], "little")
                for k in (1, 2, 3)])
            if lib.nodata_restore_guarded(*args, repl):
                raise ValueError(f"{_NAME}: the mask holds a code the guarded "
                                 f"form never writes")
        else:
            lib.nodata_restore(*args)
        out.commit(n)


class Nodata:
    """Split missing samples into a validity mask and fill their positions.

    Omit this node for tiles without missing values. Sentinel mode prevents a
    lossy stage from turning valid samples into NoData.
    """

    def __init__(self, width, value=None, dtype=None):
        self._width = int(width)
        self._dtype = None
        if value is None:
            self._pattern = None
        elif dtype is None:
            raise ValueError("give dtype alongside value, a sentinel's bit "
                             "pattern depends on the type it is read at")
        else:
            self._pattern = nodata_bits(value, dtype)
            self._dtype = dtype_code(dtype)

    def __call__(self, compressor, values_successor, mask_successor):
        succ = []
        for s in (values_successor, mask_successor):
            succ.append(s if isinstance(s, _ext.GraphID)
                        else s.parameterize(compressor))
        node = compressor.register_custom_encoder(
            _Encoder(self._width, self._pattern, self._dtype))
        return compressor.build_static_graph(node, succ, name=_NAME)
