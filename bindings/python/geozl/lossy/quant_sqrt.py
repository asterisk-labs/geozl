from dataclasses import dataclass

import numpy as np

from .._codec import quantizer
from .._dtype import dtype_code
from .._ffi import _ptr, ffi, lib

_MIN_SIDE = 10  # the block side plus the border its 3x3 mask needs


@dataclass(frozen=True)
class Noise:
    """Fitted ``variance = a + b*x`` model and diagnostics."""

    a: float
    b: float
    blocks: int
    bins: int
    range: float
    colin: float
    resid: float

    def recipe(self, max_error, *, store=None):
        """Return a SQRT recipe using this fitted curve."""
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
    """Normalize input to a list of 2-D rasters with one dtype."""
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
    """Fit ``variance = a + b*x`` to one or more rasters.

    Accepts a 2-D array, an ``(N, H, W)`` stack or a sequence of 2-D arrays.
    Raises ``RuntimeError`` when the data cannot support a fit.
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


# NULL curve. resolve reads A and B off the recipe first, so one passed here
# alongside them would be dropped. Noise.recipe is the way in.
QuantSqrt, QuantSqrtDecoder = quantizer(
    0x72D783, "geozl.lossy.quant_sqrt", "quant_sqrt", ("step", "offset"),
    "the stream holds no finite sample", extra=(ffi.NULL,))

__all__ = ["Noise", "QuantSqrt", "QuantSqrtDecoder", "fit_noise"]
