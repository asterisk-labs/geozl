import math
import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib

_CTID = 0x72D70B
_NAME = "geozl.lossless.pco_floatmult"
_MIN_SAMPLE = 10
_SAMPLE_RATIO = 40

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)


# f32/f64 constants, from pcodec data_types/float.rs
class _Spec:
    def __init__(self, ew):
        # PB precision bits, MANT mantissa digits, EO exponent bias, LB latent width
        if ew == 4:
            self.f = np.float32; self.u = np.uint32
            self.PB = 23; self.MANT = 24; self.EO = 127; self.LB = 32
        else:
            self.f = np.float64; self.u = np.uint64
            self.PB = 52; self.MANT = 53; self.EO = 1023; self.LB = 64
        self.MID = 1 << (self.LB - 1)          # sign boundary of the ordered latent
        self.MASK = (1 << self.LB) - 1
        self.MAX_FOR_SAMPLING = self.f(np.finfo(self.f).max) * self.f(0.5)  # keep pairwise products finite

# reinterpret the float's raw bytes as an unsigned integer (like f32::to_bits)
def _bits(x, S):
    return int(np.array(x, dtype=S.f).view(S.u))

# and the reverse: unsigned integer bytes back to a float
def _from_bits(b, S):
    return np.array(int(b) & S.MASK, dtype=S.u).view(S.f)[()]

# 2^power built straight from the exponent field, so it's exact
def _exp2(power, S):
    return _from_bits(((S.EO + power) << S.PB), S)

# unbiased exponent of |x|: the raw exponent field minus the format bias
def _exponent(x, S):
    return int(_bits(S.f(abs(float(x))), S) >> S.PB) - S.EO

# trailing zeros of the raw bit pattern; +0.0/-0.0 count as all bits
def _trailing_zeros(x, S):
    b = _bits(x, S)
    return S.LB if b == 0 else (b & -b).bit_length() - 1

# OBS: round half away from zero, matching Rust f32/f64::round (NOT numpy's half-even)
def _round(x, S):
    x = S.f(x)
    return S.f(np.sign(float(x))) * S.f(np.floor(S.f(abs(float(x))) + S.f(0.5)))

# monotonic bit ordering: negatives flip every bit, the rest flip the sign bit
def _to_latent_ordered(x, S):
    b = _bits(x, S)
    return ((~b) & S.MASK) if (b & S.MID) else (b ^ S.MID)

# fuzzy float comparisons from float_mult.rs (how close counts as zero / precise)
_REQUIRED_PRECISION_BITS = 6

# x scaled down below the precision we keep (its spare-mantissa-bits fraction)
def _insignificant_float_to(x, S):
    return S.f(x) * _exp2(-(S.PB - _REQUIRED_PRECISION_BITS), S)

def _is_approx_zero(small, big, S):
    return S.f(small) <= _insignificant_float_to(big, S)

# the remainder is negligible next to the value it came from
def _is_small_remainder(rem, orig, S):
    return S.f(rem) <= S.f(orig) * _exp2(-16, S)

# the value has shrunk down into its own accumulated rounding error
def _is_imprecise(value, err, S):
    return S.f(value) <= S.f(err) * _exp2(_REQUIRED_PRECISION_BITS, S)

# Euclidean GCD on two floats, carrying the accumulated rounding error so we can
# stop once the remainder stops being trustworthy. None means no clean divisor.
def _approx_pair_gcd(greater, lesser, S):
    greater, lesser = S.f(greater), S.f(lesser)
    if _is_approx_zero(lesser, greater, S) or lesser == greater:
        return None
    machine_eps = _exp2(-S.PB, S)
    g_val, g_err = greater, S.f(0.0)
    l_val, l_err = lesser, S.f(0.0)
    while True:
        prev = g_val
        ratio = _round(g_val / l_val, S)                    # times lesser fits into greater
        g_err = g_err + ratio * l_err + g_val * machine_eps  # carry the rounding error forward
        g_val = S.f(abs(float(g_val - ratio * l_val)))       # remainder after subtracting
        # remainder is ~0 or already lost in the noise: lesser is the gcd
        if _is_small_remainder(g_val, prev, S) or g_val <= g_err:
            return l_val
        # remainder too tiny or too noisy to trust: no clean gcd
        if _is_approx_zero(g_val, greater, S) or _is_imprecise(g_val, g_err, S):
            return None
        g_val, g_err, l_val, l_err = l_val, l_err, g_val, g_err  # keep greater >= lesser


def _from_inv_base(inv_base, S):
    return (S.f(1.0) / S.f(inv_base), S.f(inv_base))   # (base, inv_base)

def _from_base(base, S):
    return (S.f(base), S.f(1.0) / S.f(base))

_REQUIRED_GCD_PAIR_FREQUENCY = 0.001

# GCD of consecutive pairs, then look for one value that shows up across enough
# pairs (checked at a few percentiles of the sorted GCDs)
def _approx_sample_gcd_euclidean(sample, S):
    gcds = []
    i = 0
    while i < len(sample) - 1:                 # (0..len-1).step_by(2)
        a, b = S.f(sample[i]), S.f(sample[i + 1])
        g = _approx_pair_gcd(max(a, b), min(a, b), S)
        if g is not None:
            gcds.append(g)
        i += 2
    required = 1 + math.ceil(len(sample) * _REQUIRED_GCD_PAIR_FREQUENCY)  # min pairs that must agree
    if len(gcds) < required:
        return None
    gcds.sort()
    for pct in (0.1, 0.3, 0.5):
        cand = gcds[int(pct * len(gcds))]                 # a candidate from low/mid of the sorted gcds
        thresh = S.f(0.01) * cand
        cnt = sum(1 for g in gcds if S.f(abs(float(g - cand))) < thresh)  # how many gcds land within 1%
        if cnt >= required:
            return cand
    return None

# nudge the base to minimize the deviation from mult*base over the sample,
# weighting each sample by how many mantissa bits its mult leaves to spare
def _center_sample_base(base, sample, S):
    base = S.f(base)
    inv_base = S.f(1.0) / base
    tweak_sum = S.f(0.0)
    tweak_weight = S.f(0.0)
    for x in sample:
        x = S.f(x)
        mult = _round(x * inv_base, S)              # nearest multiple of base for this sample
        mult_exp = _exponent(mult, S)
        if mult_exp < S.PB and mult != S.f(0.0):    # skip huge mults (no precision left) and zero
            overshoot = mult * base - x             # how far mult*base misses the real value
            weight = S.f(S.PB - mult_exp)           # small mults weigh more (more bits to spare)
            tweak_sum = tweak_sum + weight * (overshoot / mult)
            tweak_weight = tweak_weight + weight
    return base - tweak_sum / tweak_weight          # shift base by the weighted-average miss

_SNAP_THRESHOLD_ABSOLUTE = 0.02
_SNAP_THRESHOLD_DECIMAL_RELATIVE = 0.01

def _round_away_f64(x):
    return math.copysign(math.floor(abs(x) + 0.5), x)

# if 1/base is nearly a whole number or a power of ten, snap to it (e.g. a base
# of 0.0999 becomes exactly 0.1); otherwise keep the base as found
def _snap_to_int_reciprocal(base, S):
    base = S.f(base)
    inv_base = S.f(1.0) / base
    round_inv_base = _round(inv_base, S)                                          # nearest whole number
    decimal_inv_base = S.f(10.0 ** _round_away_f64(math.log10(float(inv_base))))   # nearest power of ten
    if abs(float(inv_base - round_inv_base)) < _SNAP_THRESHOLD_ABSOLUTE:           # 1/base ~ integer -> e.g. 1/7
        return _from_inv_base(round_inv_base, S)
    if abs(float(inv_base - decimal_inv_base)) / float(inv_base) < _SNAP_THRESHOLD_DECIMAL_RELATIVE:  # ~ power of ten -> 0.1, 0.01
        return _from_inv_base(decimal_inv_base, S)
    return _from_base(base, S)                                                     # nothing clean, keep it

def _choose_config_by_euclidean(sample, S):
    base = _approx_sample_gcd_euclidean(sample, S)
    if base is None:
        return None
    base = _center_sample_base(base, sample, S)
    return _snap_to_int_reciprocal(base, S)

# map an integer-valued float to the unsigned latent the mult stream stores,
# keeping small integers dense and centering the sign around MID
def _int_float_to_latent(m, S):
    m = S.f(m)
    a = S.f(abs(float(m)))
    gpi = 1 << S.MANT                    # 2^mantissa_digits; above this, floats step by more than 1
    gpi_float = S.f(float(gpi))
    if a < gpi_float:
        abs_int = int(a)                 # below gpi the float already equals its integer value
    else:
        abs_int = (gpi + (_bits(a, S) - _bits(gpi_float, S))) & S.MASK  # above gpi, count representable steps
    if _bits(m, S) & S.MID:              # negative: mirror below MID (-1 keeps -0.0 apart from +0.0)
        return (S.MID - 1 - abs_int) & S.MASK
    return (S.MID + abs_int) & S.MASK    # positive: above MID

def _from_latent_numerical(l, S):
    return S.f(float(int(l)))

# int_mult's candidate-base detector, duplicated here to keep floatmult
# self-contained (faithful port of int_mult.rs choose_candidate_base)
_ZETA_OF_2 = math.pi * math.pi / 6.0
_LCB_RATIO = 1.0
_MULT_REQ = 0.5

def _im_triple_gcds(sample):
    out = []
    m = (len(sample) // 3) * 3
    for i in range(0, m, 3):
        a, b, c = sorted(sample[i:i + 3])
        g = math.gcd(int(b - a), int(c - a))  # gcd of the gaps = grid spacing this triple implies
        if g > 1:
            out.append(g)
    return out, len(sample) // 3

# entropy of a distribution with one spike of mass p and the rest spread uniformly
def _im_worst_case_entropy(p, gcd_m1):
    if gcd_m1 <= 0:
        return 0.0
    rest = (1.0 - p) / gcd_m1
    h = 0.0
    if p > 0:
        h -= p * math.log2(p)
    if rest > 0:
        h -= gcd_m1 * rest * math.log2(rest)
    return h

# root-find f(p)=0 by false position (regula falsi), nudged toward bisection for safety
def _im_false_position(f, lb, ub):
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

def _im_score_gcd(gcd, cnt, total):
    if total == 0:
        return None
    prob = cnt / total                          # observed fraction of triples with this gcd
    natural = 1.0 / (_ZETA_OF_2 * gcd * gcd)     # fraction expected by chance on uniform data
    stdev = math.sqrt(natural * (1.0 - natural) / total)
    if stdev == 0.0 or (prob - natural) / stdev < 3.0:  # z-test: reject unless well above chance
        return None
    lcb = cnt - _LCB_RATIO * math.sqrt(cnt)      # conservative lower bound on the count
    if lcb <= 0.0:
        return None
    congruence = min(_ZETA_OF_2 * lcb / total, 1.0)  # implied true probability of congruence
    gcd_m1 = gcd - 1.0
    inv_sq = 1.0 / (gcd_m1 * gcd_m1)
    f = lambda p: p ** 3 + (1.0 - p) ** 3 * inv_sq - congruence   # solve for the concentrated mass p
    p = _im_false_position(f, 1.0 / gcd, float(np.cbrt(congruence)) + 2.220446049250313e-16)
    if p is None:
        return None
    bits = math.log2(gcd) - _im_worst_case_entropy(p, gcd_m1)  # bits saved vs a flat [0, gcd)
    return bits if bits >= _MULT_REQ else None

def _choose_candidate_base(int_sample, S):
    gcds, total = _im_triple_gcds(int_sample)
    if not gcds:
        return None
    from collections import Counter
    best, best_score = None, -1.0
    for g, cnt in Counter(gcds).items():          # score each distinct gcd, keep the best
        sc = _im_score_gcd(float(g), cnt, total)
        if sc is not None and sc > best_score:
            best, best_score = g, sc
    return best

_INTERESTING_TRAILING_ZEROS = 5
_REQUIRED_TRAILING_ZEROS_FREQUENCY = 0.5
_MIN_SAMPLE = 10

# strategy for grids that are integer multiples of a power of two: find the
# largest 2^k dividing everything, pull out the integer multiples, and run the
# int-mult GCD detector on them
def _choose_config_by_trailing_zeros(sample, S):
    PB = S.PB
    def k_div(exponent, tz):
        return exponent - max(0, PB - tz)
    k = None
    count = 0
    for x in sample:
        x = S.f(x)
        tz = _trailing_zeros(x, S)
        if float(x) != 0.0 and tz >= _INTERESTING_TRAILING_ZEROS:  # ignore zero and near-noise
            kp = k_div(_exponent(x, S), tz)       # largest power of two dividing this float
            count += 1
            k = kp if k is None else min(k, kp)   # k = the power of two dividing ALL of them
    required = max(math.ceil(len(sample) * _REQUIRED_TRAILING_ZEROS_FREQUENCY), _MIN_SAMPLE)
    if count < required:
        return None
    int_sample = []
    lshift = S.LB - PB - 1
    explicit_mantissa = S.MID
    for x in sample:
        x = S.f(x)
        exponent = _exponent(x, S)
        kp = k_div(exponent, _trailing_zeros(x, S))
        if kp >= k and exponent < k + S.LB:
            # rebuild x/2^k as an integer: restore the implicit mantissa bit, then
            # shift so the units place lands at the bottom
            rshift = S.LB - 1 - (exponent - k)
            lshifted = ((_bits(x, S) << lshift) & S.MASK) | explicit_mantissa
            int_sample.append(lshifted >> rshift)
    if len(int_sample) >= required:
        cand = _choose_candidate_base(int_sample, S)   # integer grid step among the multiples
        int_base = cand if cand is not None else 1
        base = _from_latent_numerical(int_base, S) * _exp2(k, S)  # scale back by the 2^k we pulled out
        return _from_base(base, S)
    return None

_CLASSIC_MEMORIZABLE_BINS = 256

def _est_bits_saved_per_num(sample, primary_fn):
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0.0])
    for x in sample:
        primary, bits = primary_fn(x)
        e = acc[primary]
        e[0] += 1
        e[1] += bits
    # only count savings from primaries that are rare (a frequent primary the
    # entropy stage would just memorize, so it isn't really a win here)
    cutoff = max(1, int(len(sample) / _CLASSIC_MEMORIZABLE_BINS))
    total = sum(bits for cnt, bits in acc.values() if cnt <= cutoff)
    return total / len(sample)

# the gate: estimate bits saved per number against storing the raw float,
# modelling the adjustment cost as ~1 + 2*log2|adj|. None if it doesn't clear
# the required margin
def _bits_saved_per_num_over_classic(config, sample, S):
    base, inv_base = config
    PB = S.PB

    def primary_fn(x):
        x = S.f(x)
        mult = _round(x * inv_base, S)
        primary = _int_float_to_latent(mult, S)
        exp_i = _exponent(mult, S)
        inter_base_bits = 0 if (exp_i < 0 or exp_i >= PB) else (PB - exp_i)  # bits the grid removes
        approx_unsigned = _to_latent_ordered(mult * base, S)
        x_unsigned = _to_latent_ordered(x, S)
        abs_adj = abs(x_unsigned - approx_unsigned)          # ULP gap between x and mult*base
        adj_bits = 1 + 2 * abs_adj.bit_length()              # modelled cost of storing that gap
        return primary, float(inter_base_bits) - float(adj_bits)  # net bits saved for this number

    v = _est_bits_saved_per_num(sample, primary_fn)
    return v if v >= _MULT_REQ else None

# try the power-of-two strategy first, fall back to the Euclidean one
def _choose_config(sample, S):
    cfg = _choose_config_by_trailing_zeros(sample, S)
    if cfg is not None:
        return cfg
    return _choose_config_by_euclidean(sample, S)


# pcodec choose_sample + filter_sample: normal floats, abs value, in range
def _sample_euclidean(values, S):
    v = np.asarray(values, S.f)
    tiny = np.finfo(S.f).tiny
    keep = np.isfinite(v) & (np.abs(v) >= tiny) & (np.abs(v) <= S.MAX_FOR_SAMPLING)
    finite = np.abs(v[keep])
    n = finite.size
    if n < _MIN_SAMPLE:
        return None
    k = _MIN_SAMPLE + (n - _MIN_SAMPLE) // _SAMPLE_RATIO
    return finite[np.random.default_rng(0).choice(n, size=min(k, n), replace=False)]


# cheap fallback heuristic, not from pcodec
_REQUIRED_BITS_SAVED_PER_NUM = 1.0
_CANDIDATES = [10.0 ** -d for d in range(1, 8)] + [2.0 ** -b for b in range(1, 14)]


def _sample_candidates(x):
    n = x.size
    if n < _MIN_SAMPLE:
        return None
    k = _MIN_SAMPLE + (n - _MIN_SAMPLE) // _SAMPLE_RATIO
    return x[np.random.default_rng(0).choice(n, size=min(k, n), replace=False)]


def _approx_entropy_bits(vals):
    if vals.size == 0:
        return 0.0
    _, counts = np.unique(vals, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _choose_base_candidates(values, elt_width):
    smp = _sample_candidates(values)
    if smp is None:
        return None
    finite = smp[np.isfinite(smp)]
    if finite.size < _MIN_SAMPLE:
        return None
    # baseline cost: byte-plane entropy of the raw floats
    raw = np.ascontiguousarray(finite).view(np.uint8).reshape(-1, elt_width)
    base_bits = sum(_approx_entropy_bits(raw[:, b]) for b in range(elt_width))
    uint = np.uint32 if elt_width == 4 else np.uint64
    best_base, best_saved = None, 0.0
    for base in _CANDIDATES:
        mult = np.round(finite / base)
        recon = (mult * base).astype(finite.dtype)
        # cost with this base: entropy of the mult stream plus the x-vs-recon bit diff
        mu = np.ascontiguousarray(mult.astype(finite.dtype)).view(np.uint8).reshape(-1, elt_width)
        diff = (np.ascontiguousarray(finite).view(uint)
                ^ np.ascontiguousarray(recon).view(uint)).view(np.uint8).reshape(-1, elt_width)
        cand_bits = (sum(_approx_entropy_bits(mu[:, b]) for b in range(elt_width))
                     + sum(_approx_entropy_bits(diff[:, b]) for b in range(elt_width)))
        saved = base_bits - cand_bits
        if saved > best_saved:
            best_saved, best_base = saved, base
    if best_base is None or best_saved < _REQUIRED_BITS_SAVED_PER_NUM:
        return None
    return float(best_base)


# method="euclidean" is the faithful pcodec detector; "candidates" is the cheap
# byte-plane heuristic. Returns a float base the data lies on, or None.
def choose_base(values, elt_width, *, method="euclidean"):
    if method == "candidates":
        return _choose_base_candidates(values, elt_width)
    if method != "euclidean":
        raise ValueError(f"unknown method: {method!r}")
    S = _Spec(elt_width)
    sample = _sample_euclidean(values, S)
    if sample is None:
        return None
    cfg = _choose_config(sample, S)          # (base, inv_base) or None
    if cfg is None:
        return None
    if _bits_saved_per_num_over_classic(cfg, sample, S) is None:  # base must clear the gate
        return None
    return float(cfg[0])


class _Encoder(_ext.CustomEncoder):
    def __init__(self, base):
        super().__init__()
        self._base = float(base)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        prim = state.create_output(0, n, elt)
        sec = state.create_output(1, n, elt)
        lib.floatmult_split(_ptr(prim.mut_content.as_nparray()),
                            _ptr(sec.mut_content.as_nparray()),
                            _ptr(inp.content.as_nparray()), n, elt,
                            ffi.cast("double", self._base),
                            ffi.cast("double", 1.0 / self._base))
        state.send_codec_header(struct.pack("<d", self._base))
        prim.commit(n)
        sec.commit(n)


class FloatMultDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        prim = state.singleton_inputs[0]
        sec = state.singleton_inputs[1]
        elt = prim.elt_width
        if sec.elt_width != elt or prim.num_elts != sec.num_elts:
            raise ValueError(f"{_NAME}: corrupt stream shape")
        raw = bytes(state.codec_header)
        if len(raw) != 8:
            raise ValueError(f"{_NAME}: bad codec header size")
        base = struct.unpack("<d", raw)[0]
        if base == 0.0 or base != base:  # zero or NaN
            raise ValueError(f"{_NAME}: invalid base")
        n = prim.num_elts
        out = state.create_output(0, n, elt)
        bad = lib.floatmult_join(_ptr(out.mut_content.as_nparray()),
                                 _ptr(prim.content.as_nparray()),
                                 _ptr(sec.content.as_nparray()), n, elt,
                                 ffi.cast("double", base))
        if bad:
            raise ValueError(f"{_NAME}: corrupt frame")
        out.commit(n)


# split node against a float base; successors receive (mults, adjustments)
class FloatMult:

    def __init__(self, base):
        if base == 0.0 or base != base:
            raise ValueError("floatmult base must be nonzero and finite")
        self._base = float(base)

    def __call__(self, compressor, mults_successor, adjs_successor):
        succ = []
        for s in (mults_successor, adjs_successor):
            if not isinstance(s, _ext.GraphID):
                s = s.parameterize(compressor)
            succ.append(s)
        node = compressor.register_custom_encoder(_Encoder(self._base))
        return compressor.build_static_graph(node, succ, name=_NAME)