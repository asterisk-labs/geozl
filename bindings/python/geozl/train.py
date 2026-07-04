"""Fit the wp_static kernel for a band at encode time. Not a codec, it runs on
the encode side and never touches decode. Pure numpy."""

import numpy as np

_PLANAR = ((1, -1, 0, 0), 0)
_I16_MIN, _I16_MAX = -32768, 32767


def _fit_region(band):
    b = np.ascontiguousarray(band).astype(np.int64)
    x = b[2:, 1:-1]
    return {
        "target": (x - b[2:, :-2]).ravel().astype(np.float64),
        "A": np.stack([b[1:-1, 1:-1].ravel(), b[1:-1, :-2].ravel(),
                       b[1:-1, 2:].ravel(), b[:-2, 1:-1].ravel()],
                      axis=1).astype(np.float64),
    }


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


def _entropy_bits(resid):
    stored = resid.astype(np.int64) & 0xFFFF
    _, counts = np.unique(stored, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _solve(A, target, method, irls_iters, eps):
    a, *_ = np.linalg.lstsq(A, target, rcond=None)
    if method == "l1":
        for _ in range(irls_iters):
            w = 1.0 / np.maximum(np.abs(A @ a - target), eps)
            a, *_ = np.linalg.lstsq(A * w[:, None], target * w, rcond=None)
    return a


def fit_wp_static(band, *, shift_max=8, method="l2", irls_iters=5, eps=1.0):
    """Return (coeffs, shift) for geozl.lossless.WpStaticInt, the four int16
    coefficients in the order N, NW, NE, NN and the shift. Every candidate is
    scored with the exact codec residual over the whole band, and the result is
    never worse than planar."""
    band = np.asarray(band)
    if band.ndim != 2 or band.shape[0] < 3 or band.shape[1] < 3:
        return _PLANAR

    region = _fit_region(band)
    a = _solve(region["A"], region["target"], method, irls_iters, eps)

    best_cost = _entropy_bits(_residual(band, *_PLANAR))
    best = _PLANAR
    for shift in range(shift_max + 1):
        q = np.round(a * (1 << shift)).astype(np.int64)
        if q.min() < _I16_MIN or q.max() > _I16_MAX:
            continue
        cost = _entropy_bits(_residual(band, tuple(int(v) for v in q), shift))
        if cost < best_cost:
            best_cost, best = cost, (tuple(int(v) for v in q), shift)

    return best