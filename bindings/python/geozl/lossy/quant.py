import math
import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib
from ._base import require_checksum_disabled

_CTID = 0x72D780
_NAME = "geozl.lossy.quant"

# Mirrors the quant_dtype enum in quant_dtype.h, the value is the wire code.
_Q_DTYPE = {
    np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2,
    np.dtype("uint64"): 3, np.dtype("int8"): 4, np.dtype("int16"): 5,
    np.dtype("int32"): 6, np.dtype("int64"): 7, np.dtype("float16"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10,
}

# Element width per dtype code, derived from the table above so it cannot drift
# from it. Codes are 0..N-1, so a tuple indexes straight by code.
_WIDTH = tuple(dt.itemsize
               for dt, _c in sorted(_Q_DTYPE.items(), key=lambda kv: kv[1]))


def dtype_code(dtype):
    """The quant_dtype wire code for dtype, or None when quant has no kernel
    for it. Keyed on the numpy dtype, so a byte-swapped array is refused rather
    than quantized as native."""
    return _Q_DTYPE.get(np.dtype(dtype))

# uint8 dtype, uint8 curve, uint8 flags, then step and offset as IEEE
# doubles and nsub as a uint64, little endian.
_HEADER = struct.Struct("<BBBddQ")

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric],
)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, spec, params, dtype):
        super().__init__()
        self._sp = spec
        self._p = params
        self._dtype = dtype

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        require_checksum_disabled(state, _NAME)
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        if _WIDTH[self._dtype] != elt:
            raise ValueError(
                f"{_NAME}: dtype {self._dtype} does not match {elt}-byte samples")
        out = state.create_output(0, n, elt)
        # quant_fit tightens the step until the round trip holds the declared
        # bound, so the header takes its parameters from the copy it worked on
        # and not from the resolved ones, which are reused for the next tile.
        p = ffi.new("quant_params*")
        p.curve, p.flags, p.step, p.offset, p.nsub = (
            self._p.curve, self._p.flags, self._p.step, self._p.offset,
            self._p.nsub)
        chk = np.empty(n * elt, dtype=np.uint8)
        fit = lib.quant_fit(_ptr(out.mut_content.as_nparray()), _ptr(chk),
                            _ptr(inp.content.as_nparray()), self._sp, p,
                            self._dtype, n)
        if fit < 0:
            raise ValueError(
                f"{_NAME}: resolved parameters its own kernels reject")
        if fit != 0:
            raise ValueError(
                f"{_NAME}: cannot hold its declared error on this tile")
        state.send_codec_header(_HEADER.pack(
            self._dtype, p.curve, p.flags, p.step, p.offset, p.nsub))
        out.commit(n)


class QuantDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        inp = state.singleton_inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        dtype, curve, flags, step, offset, nsub = _HEADER.unpack(
            bytes(state.codec_header))
        if not 0 <= dtype < len(_WIDTH) or _WIDTH[dtype] != elt:
            raise ValueError(f"{_NAME}: bad dtype in codec header")
        if curve > 2:
            raise ValueError(f"{_NAME}: bad curve in codec header")
        if flags & ~(_STORE_VALUES | _NONNEGATIVE):
            raise ValueError(f"{_NAME}: bad flags in codec header")
        if not (math.isfinite(step) and math.isfinite(offset)) or step < 0.0:
            raise ValueError(f"{_NAME}: bad step in codec header")
        if flags & _STORE_VALUES and (curve != 0 or dtype > _LAST_INT_CODE):
            raise ValueError(f"{_NAME}: a stored reconstruction needs the "
                             f"linear curve on an integer type")
        if curve == 2 and not offset > 0.0:
            raise ValueError(f"{_NAME}: the log curve needs a positive anchor")
        if curve != 2 and nsub != 0:
            raise ValueError(f"{_NAME}: nsub is the log curve's exact region")
        p = ffi.new("quant_params*")
        p.curve, p.flags, p.step, p.offset, p.nsub = (
            curve, flags, step, offset, nsub)
        out = state.create_output(0, n, elt)
        if lib.quant_decode(_ptr(out.mut_content.as_nparray()),
                            _ptr(inp.content.as_nparray()), p, dtype, n) != 0:
            raise ValueError(f"{_NAME}: the kernel refused this codec header")
        out.commit(n)


# Codes 0..7 are the integer types, 8..10 the floats.
_LAST_INT_CODE = 7

# Bit 0, the stream already holds the reconstruction. Bit 1, the encoder scanned
# the tile and found nothing negative, so the decoder floors at zero. These
# mirror QUANT_FLAG_* in quant_params.h and the guards below mirror the ones in
# decode_quant_binding.c, so a bit added on one side has to be added on both.
_STORE_VALUES = 1
_NONNEGATIVE = 2


class Quant:
    """Head of graph lossy node. Quantizes uniformly in a warped domain, so the
    error it holds follows the curve rather than being flat. Disable
    ContentChecksum on the CCtx, the round trip is not bit exact.

    error is the recipe the 2d API takes, "abs:V", "rel:P%" or
    "shot:a=A,b=B,k=K". The log curve anchors its grid on the smallest magnitude
    in the tile, so the tile has to be handed over here rather than at encode.
    """

    def __init__(self, error, tile):
        arr = np.ascontiguousarray(tile)
        code = dtype_code(arr.dtype)
        if code is None:
            raise ValueError(f"quant does not support dtype {arr.dtype!r}")
        if not isinstance(error, str) or not error:
            raise ValueError(f"error must be a recipe string, got {error!r}")

        err = ffi.new("char[]", 256)
        sp = ffi.new("quant_spec*")
        if lib.quant_spec_parse(error.encode("utf-8"), sp, err, len(err)) != 0:
            raise ValueError(ffi.string(err).decode("utf-8", "replace"))
        lo, hi = ffi.new("double*"), ffi.new("double*")
        neg = ffi.new("int*")
        lib.quant_scan(_ptr(arr), code, arr.size, lo, hi, neg)
        p = ffi.new("quant_params*")
        if lib.quant_spec_resolve(sp, code, lo[0], hi[0], neg[0], p, err,
                                  len(err)) != 0:
            raise ValueError(ffi.string(err).decode("utf-8", "replace"))
        self._sp = sp
        self._p = p
        self._dtype = code

    def __call__(self, compressor, successor):
        if not isinstance(successor, _ext.GraphID):
            successor = successor.parameterize(compressor)
        node = compressor.register_custom_encoder(
            _Encoder(self._sp, self._p, self._dtype))
        return compressor.build_static_graph(node, [successor], name=_NAME)