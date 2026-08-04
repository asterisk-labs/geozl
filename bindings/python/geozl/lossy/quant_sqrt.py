from dataclasses import dataclass

import numpy as np

from .._codec import quantizer
from .._dtype import dtype_code
from .._ffi import _ptr, ffi, lib

_MIN_SIDE = 10  # the block side plus the border its 3x3 mask needs


@dataclass(frozen=True)
class Noise:
    """a and b are the curve. The rest says how far to believe it, and the fit
    reports rather than decides, since what counts as enough depends on how the
    rasters were chosen. colin is the one that catches a bad fit: large means a
    and b are collinear and the split between them is arbitrary."""

    a: float
    b: float
    blocks: int
    bins: int
    range: float
    colin: float
    resid: float

    def recipe(self, max_error, *, store=None):
        """max_error sigmas of this curve, at full precision so it resolves to
        the grid the fit produced and not to a nearby one."""
        if not max_error > 0:
            raise ValueError(f"max_error must be positive, got {max_error!r}")
        s = f"SQRT:MAX_ERROR={max_error:.17g}N,A={self.a:.17g},B={self.b:.17g}"
        if store is not None:
            if store not in ("INDEX", "VALUES"):
                raise ValueError(f"store is INDEX or VALUES, got {store!r}")
            s += f",STORE={store}"
        return s

    def __str__(self):
        return (f"sigma^2 = {self.a:.4g} + {self.b:.4g}*x  "
                f"[{self.blocks} blocks, {self.bins}/24 bins, "
                f"range {self.range:.1f}x, colin {self.colin:.1f}, "
                f"resid {self.resid:.3f}]")


def _rasters(data):
    """A 3d array is a stack over its first axis. Shapes may differ, dtypes may
    not, since the fit reads raw samples through one code."""
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            out = [data]
        elif data.ndim == 3:
            out = [data[i] for i in range(data.shape[0])]
        else:
            raise ValueError(
                f"give a 2d raster or a 3d stack, got {data.ndim} dimensions")
    else:
        out = [np.asarray(r) for r in data]
        if not out:
            raise ValueError("no rasters to fit against")
        for r in out:
            if r.ndim != 2:
                raise ValueError(
                    f"every raster in a sequence is 2d, got {r.ndim} dimensions")

    dt = out[0].dtype
    for r in out:
        if r.dtype != dt:
            raise ValueError(
                f"the rasters have to share a dtype, got {dt} and {r.dtype}")
    return [np.ascontiguousarray(r) for r in out], dt


def fit_noise(data):
    """Fit sigma**2 = a + b*x over a 2d raster, a 3d stack read as (N, H, W), or
    any sequence of 2d arrays. Everything pools into one curve.

    Scatterplot of local mean against local variance, low quantile per intensity
    bin, after Abramova and others, SPIE 10004, 2016. Raises RuntimeError when
    the pool does not support a curve, which is the answer for rasters that are
    flat or cover too little of their range. Pool more of the product rather
    than falling back to one tile.
    """
    rasters, dt = _rasters(data)
    code = dtype_code(dt)
    if code is None:
        raise ValueError(f"the noise fit does not support dtype {dt}")
    for r in rasters:
        h, w = r.shape
        if h < _MIN_SIDE or w < _MIN_SIDE:
            raise ValueError(
                f"a raster of {h}x{w} is too small to measure blocks on, the "
                f"fit needs at least {_MIN_SIDE} samples on a side")

    acc = ffi.new("quant_sqrt_accum*")
    lib.quant_sqrt_accum_init(acc)
    try:
        for r in rasters:
            h, w = r.shape
            if lib.quant_sqrt_accum_push(acc, _ptr(r), code, w, h) != 0:
                raise RuntimeError(
                    f"the noise fit could not measure a {h}x{w} raster")
        out = ffi.new("quant_sqrt_noise*")
        err = ffi.new("char[]", 256)
        if lib.quant_sqrt_accum_solve(acc, out, err, len(err)) != 0:
            reason = ffi.string(err).decode("utf-8", "replace")
            raise RuntimeError(f"the noise fit found no curve: {reason}")
        return Noise(a=out.a, b=out.b, blocks=out.blocks, bins=out.bins,
                     range=out.range, colin=out.colin, resid=out.resid)
    finally:
        lib.quant_sqrt_accum_free(acc)


def _plan(spec, dtype, src, nb_elts, params):
    err = ffi.new("char[]", 256)
    stats = ffi.new("quant_sqrt_stats*")
    if lib.quant_sqrt_scan(src, dtype, nb_elts, stats):
        return "the stream holds no finite sample"
    # NULL curve. resolve reads A and B off the recipe first, so one passed
    # here alongside them would be dropped. Noise.recipe is the way in.
    if lib.quant_sqrt_resolve(spec, dtype, stats, ffi.NULL, params, err,
                              len(err)):
        return ffi.string(err).decode("utf-8", "replace")
    return None


QuantSqrt, QuantSqrtDecoder = quantizer(
    0x72D783, "geozl.lossy.quant_sqrt", "quant_sqrt", ("step", "offset"),
    _plan)

__all__ = ["Noise", "QuantSqrt", "QuantSqrtDecoder", "fit_noise"]
