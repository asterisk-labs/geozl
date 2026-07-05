"""Numpy reference fit for the wp_static kernel, the oracle the tests check the
C fit against. Scores candidates by byte plane entropy, the way the entropy
stage does, not by a 16 bit alphabet."""

import numpy as np

_PLANAR = ((1, -1, 0, 0), 0)
_I16_MIN, _I16_MAX = -32768, 32767


def _residual(band, coeffs, shift):
    x = np.ascontiguousarray(band).astype(np.int64)
    cN, cNW, cNE, cNN = (int(c) for c in coeffs)
    rnd = (1 << (shift - 1)) if shift else 0
    N = np.zeros_like(x);  N[1:, :] = x[:-1, :]
    NW = np.zeros_like(x); NW[1:, 1:] = x[:-1, :-1]
    NE = np.zeros_like(x); NE[1:, :-1] = x[:-1, 1:]
    NN = np.zeros_like(x); NN[2:, :] = x[:-2, :]
    W = np.zeros_like(x);  W[:, 1:] = x[:, :-1]
    K = (cN * N + cNW * NW + cNE * NE + cNN * NN + rnd) >> shift
    return x - (W + K)


def _plane_bits(resid):
    r16 = (resid.astype(np.int64) & 0xFFFF).astype(np.int16).astype(np.int64)
    zz = ((r16 << 1) ^ (r16 >> 63)) & 0xFFFF
    hi = (zz >> 8) & 0xFF
    lo = zz & 0xFF

    def H(plane):
        _, counts = np.unique(plane, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    return H(hi) + H(lo)


def _fit_region(band):
    b = np.ascontiguousarray(band).astype(np.int64)
    x = b[2:, 1:-1]
    return {
        "target": (x - b[2:, :-2]).ravel().astype(np.float64),
        "A": np.stack([b[1:-1, 1:-1].ravel(), b[1:-1, :-2].ravel(),
                       b[1:-1, 2:].ravel(), b[:-2, 1:-1].ravel()],
                      axis=1).astype(np.float64),
    }


def fit_wp_static(band, *, shift_max=8):
    band = np.asarray(band)
    if band.ndim != 2 or band.shape[0] < 3 or band.shape[1] < 3:
        return _PLANAR

    region = _fit_region(band)
    a, *_ = np.linalg.lstsq(region["A"], region["target"], rcond=None)

    best_cost = _plane_bits(_residual(band, *_PLANAR))
    best = _PLANAR
    for shift in range(shift_max + 1):
        q = np.round(a * (1 << shift)).astype(np.int64)
        if q.min() < _I16_MIN or q.max() > _I16_MAX:
            continue
        cost = _plane_bits(_residual(band, tuple(int(v) for v in q), shift))
        if cost < best_cost:
            best_cost, best = cost, (tuple(int(v) for v in q), shift)

    return best