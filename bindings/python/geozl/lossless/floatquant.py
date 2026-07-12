import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib

_CTID = 0x72D70A
_NAME = "geozl.lossless.pco_floatquant"
_QUANT_REQUIRED_BITS_SAVED_PER_NUM = 1.5
_MIN_SAMPLE = 10
_SAMPLE_RATIO = 40

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)


def _prec_bits(elt_width):
    return 23 if elt_width == 4 else 52


def _single_cat_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p)


def _worst_case_categorical_entropy(p, n_m1):
    if n_m1 <= 0:
        return _single_cat_entropy(p)
    return _single_cat_entropy(p) + n_m1 * _single_cat_entropy((1.0 - p) / n_m1)


def _sample(u):
    n = u.size
    if n < _MIN_SAMPLE:
        return None
    k = _MIN_SAMPLE + (n - _MIN_SAMPLE) // _SAMPLE_RATIO
    rng = np.random.default_rng(0)
    return u[rng.choice(n, size=min(k, n), replace=False)]


def _trailing_zeros(bits, prec):
    """Trailing-zero count of each bit pattern, capped at prec. 0 yields prec."""
    tz = np.zeros(bits.shape, np.int64)
    nz = bits != 0
    if np.any(nz):
        v = bits[nz]
        cnt = np.zeros(v.shape, np.int64)
        shift = v.copy()
        for _ in range(prec):
            iszero = (shift & 1) == 0
            done = (shift == 0)
            step = iszero & ~done
            cnt[step] += 1
            shift[step] >>= 1
        tz[nz] = np.minimum(cnt, prec)
    tz[~nz] = prec
    return tz


def choose_k(values, elt_width, *, required_bits_saved=_QUANT_REQUIRED_BITS_SAVED_PER_NUM):
    """Return the best number of quantized mantissa bits k >= 1, or None. values
    is a 1D unsigned numpy array of the float bit patterns."""
    smp = _sample(values)
    if smp is None:
        return None
    prec = _prec_bits(elt_width)
    tz = _trailing_zeros(smp.astype(object), prec)  # exact big-int trailing zeros
    hist = np.zeros(prec + 1, np.int64)
    for t in tz:
        hist[int(t)] += 1
    # reverse cumulative: at index k, count with >= k trailing zeros
    rev = np.cumsum(hist[::-1])[::-1]
    n = float(smp.size)
    best_k, best_saved = 0, 0.0
    for k in range(1, prec + 1):
        occ = int(rev[k])
        if occ == 0:
            continue
        freq = occ / n
        n_cat_m1 = float((1 << k) - 1)
        wc = _worst_case_categorical_entropy(freq, n_cat_m1)
        saved = k - wc
        if saved > best_saved:
            best_k, best_saved = k, saved
        else:
            break  # take the first local max, later k make (x>>k) degenerate
    if best_k == 0:
        return None
    # gate: verify bits actually saved per num on the sample
    quot = (smp >> np.uint64(best_k)) if elt_width == 8 else (smp.astype(np.uint32) >> np.uint32(best_k))
    from collections import Counter
    counts = Counter(int(x) for x in quot)
    cutoff = max(1, smp.size // 256)
    infrequent = sum(c for c in counts.values() if c <= cutoff)
    if (infrequent / n) * best_saved <= required_bits_saved:
        return None
    return int(best_k)



class _Encoder(_ext.CustomEncoder):
    def __init__(self, k):
        super().__init__()
        self._k = int(k)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        prim = state.create_output(0, n, elt)
        sec = state.create_output(1, n, elt)
        lib.floatquant_split(_ptr(prim.mut_content.as_nparray()),
                             _ptr(sec.mut_content.as_nparray()),
                             _ptr(inp.content.as_nparray()), n, elt,
                             ffi.cast("unsigned", self._k))
        state.send_codec_header(struct.pack("<B", self._k))
        prim.commit(n)
        sec.commit(n)


class FloatQuantDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        prim = state.singleton_inputs[0]
        sec = state.singleton_inputs[1]
        elt = prim.elt_width
        if sec.elt_width != elt or prim.num_elts != sec.num_elts:
            raise ValueError(f"{_NAME}: corrupt stream shape")
        raw = bytes(state.codec_header)
        if len(raw) != 1:
            raise ValueError(f"{_NAME}: bad codec header size")
        k = raw[0]
        if k < 1 or k > _prec_bits(elt):
            raise ValueError(f"{_NAME}: k out of range")
        n = prim.num_elts
        out = state.create_output(0, n, elt)
        bad = lib.floatquant_join(_ptr(out.mut_content.as_nparray()),
                                  _ptr(prim.content.as_nparray()),
                                  _ptr(sec.content.as_nparray()), n, elt,
                                  ffi.cast("unsigned", k))
        if bad:
            raise ValueError(f"{_NAME}: corrupt frame")
        out.commit(n)


class FloatQuant:
    """Split node for quantized-mantissa floats. Successors receive (high, low-k)."""

    def __init__(self, k):
        if k < 1:
            raise ValueError("floatquant k must be >= 1")
        self._k = int(k)

    def __call__(self, compressor, primary_successor, secondary_successor):
        succ = []
        for s in (primary_successor, secondary_successor):
            if not isinstance(s, _ext.GraphID):
                s = s.parameterize(compressor)
            succ.append(s)
        node = compressor.register_custom_encoder(_Encoder(self._k))
        return compressor.build_static_graph(node, succ, name=_NAME)