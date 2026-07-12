import math
import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib


def ctypes_u64(v):
    return ffi.cast("uint64_t", int(v))

_CTID = 0x72D709
_NAME = "geozl.lossless.pco_intmult"

# pcodec constants (constants.rs / int_mult.rs)
_ZETA_OF_2 = math.pi * math.pi / 6.0
_LCB_RATIO = 1.0
_MIN_SAMPLE = 10
_SAMPLE_RATIO = 40
_MULT_REQUIRED_BITS_SAVED_PER_NUM = 0.5  # MULT_REQUIRED_BITS_SAVED_PER_NUM

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)


def _sample(u):
    """Deterministic sample without replacement, ~1 in SAMPLE_RATIO, min 10."""
    n = u.size
    if n < _MIN_SAMPLE:
        return None
    k = _MIN_SAMPLE + (n - _MIN_SAMPLE) // _SAMPLE_RATIO
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(k, n), replace=False)
    return u[idx].astype(object)  # Python ints, exact GCD arithmetic


def _triple_gcds(sample):
    """GCD of (b-a, c-a) for each sorted triple, keeping those > 1."""
    m = (len(sample) // 3) * 3
    out = []
    for i in range(0, m, 3):
        a, b, c = sorted(sample[i:i + 3])
        g = math.gcd(int(b - a), int(c - a))
        if g > 1:
            out.append(g)
    return out, len(sample) // 3


def _worst_case_entropy(p, gcd_m1):
    """Entropy of a categorical with one mass p and gcd_m1 equal masses."""
    if gcd_m1 <= 0:
        return 0.0
    rest = (1.0 - p) / gcd_m1
    h = 0.0
    if p > 0:
        h -= p * math.log2(p)
    if rest > 0:
        h -= gcd_m1 * rest * math.log2(rest)
    return h


def _false_position(f, lb, ub):
    tol = 1e-4
    flb, fub = f(lb), f(ub)
    if flb > 0.0 or fub < 0.0:
        return None
    while ub - lb > tol and fub - flb > 0.0:
        w = 0.001 + 0.998 * fub / (fub - flb)
        mid = w * lb + (1.0 - w) * ub
        fmid = f(mid)
        if fmid < 0.0:
            lb, flb = mid, fmid
        else:
            ub, fub = mid, fmid
    return 0.5 * (lb + ub)


def _score_gcd(gcd, triples_w_gcd, total_triples, required_bits_saved):
    """z test + LCB + bits-saved bound, port of filter_score_triple_gcd."""
    if total_triples == 0:
        return None
    prob = triples_w_gcd / total_triples
    natural = 1.0 / (_ZETA_OF_2 * gcd * gcd)
    stdev = math.sqrt(natural * (1.0 - natural) / total_triples)
    if stdev == 0.0:
        return None
    if (prob - natural) / stdev < 3.0:  # z test
        return None
    lcb = triples_w_gcd - _LCB_RATIO * math.sqrt(triples_w_gcd)
    if lcb <= 0.0:
        return None
    congruence = min(_ZETA_OF_2 * lcb / total_triples, 1.0)
    gcd_m1 = gcd - 1.0
    inv_sq = 1.0 / (gcd_m1 * gcd_m1)
    f = lambda p: p ** 3 + (1.0 - p) ** 3 * inv_sq - congruence
    lb = 1.0 / gcd
    ub = float(np.cbrt(congruence)) + 2.220446049250313e-16  # cbrt + f64 EPSILON
    p = _false_position(f, lb, ub)
    if p is None:
        return None
    bits_saved = math.log2(gcd) - _worst_case_entropy(p, gcd_m1)
    if bits_saved < required_bits_saved:
        return None
    return bits_saved


def _est_bits_saved(sample, base):
    """Fraction of the sample whose quotient is infrequent, per
    est_bits_saved_per_num's memorizable-bins gate."""
    from collections import Counter
    quots = Counter(int(x) // base for x in sample)
    # CLASSIC_MEMORIZABLE_BINS_LOG default is 8 -> 256 bins
    cutoff = max(1, len(sample) // 256)
    infrequent = sum(c for c in quots.values() if c <= cutoff)
    return infrequent / len(sample)


def choose_base(values, *, required_bits_saved=_MULT_REQUIRED_BITS_SAVED_PER_NUM,
                skip_pow2=True):
    """Integer base >= 2 the data are multiples of, or None; values is a 1D
    unsigned array. required_bits_saved is the acceptance threshold; skip_pow2
    drops pow2 bases (redundant with the byte-plane stage)."""
    smp = _sample(values)
    if smp is None:
        return None
    gcds, total = _triple_gcds(smp)
    if not gcds:
        return None
    from collections import Counter
    counts = Counter(gcds)
    best, best_score = None, -1.0
    for g, cnt in counts.items():
        sc = _score_gcd(float(g), cnt, total, required_bits_saved)
        if sc is not None and sc > best_score:
            best, best_score = g, sc
    if best is None:
        return None
    if skip_pow2 and (best & (best - 1)) == 0:
        return None
    if _est_bits_saved(smp, best) * best_score <= required_bits_saved:
        return None
    return int(best)


class _Encoder(_ext.CustomEncoder):
    def __init__(self, base):
        super().__init__()
        self._base = base

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        mults = state.create_output(0, n, elt)
        adjs = state.create_output(1, n, elt)
        lib.intmult_split(_ptr(mults.mut_content.as_nparray()),
                          _ptr(adjs.mut_content.as_nparray()),
                          _ptr(inp.content.as_nparray()), n, elt,
                          ctypes_u64(self._base))
        state.send_codec_header(int(self._base).to_bytes(8, "little")[:elt])
        mults.commit(n)
        adjs.commit(n)


class IntMultDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        mults = state.singleton_inputs[0]
        adjs = state.singleton_inputs[1]
        elt = mults.elt_width
        if adjs.elt_width != elt or mults.num_elts != adjs.num_elts:
            raise ValueError(f"{_NAME}: corrupt stream shape")
        raw = bytes(state.codec_header)
        if len(raw) != elt:
            raise ValueError(f"{_NAME}: bad codec header size")
        base = int.from_bytes(raw, "little")
        if base < 2:
            raise ValueError(f"{_NAME}: base below 2")
        n = mults.num_elts
        out = state.create_output(0, n, elt)
        bad = lib.intmult_join(_ptr(out.mut_content.as_nparray()),
                               _ptr(mults.content.as_nparray()),
                               _ptr(adjs.content.as_nparray()), n, elt,
                               ctypes_u64(base))
        if bad:
            raise ValueError(f"{_NAME}: corrupt frame")
        out.commit(n)


class IntMult:
    """Split node against an explicit base. Successors receive (mults, adjs)."""

    def __init__(self, base):
        if base < 2:
            raise ValueError("intmult base must be >= 2")
        self._base = int(base)

    def __call__(self, compressor, mults_successor, adjs_successor):
        succ = []
        for s in (mults_successor, adjs_successor):
            if not isinstance(s, _ext.GraphID):
                s = s.parameterize(compressor)
            succ.append(s)
        node = compressor.register_custom_encoder(_Encoder(self._base))
        return compressor.build_static_graph(node, succ, name=_NAME)