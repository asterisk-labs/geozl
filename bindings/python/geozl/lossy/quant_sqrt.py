"""quant_sqrt, and the sensor noise curve it quantizes against.

The codec holds a bound in units of the noise, sigma**2 = a + b*x. Give it a and
b in the recipe and the grid is fixed before any data is read, so every tile of
the product lands on the same levels. Leave them out and the curve is fitted
from whatever raster is being compressed, which is a fallback and not the
intended path.

fit_noise is how the curve gets measured once. Feed it the rasters, keep the two
numbers, put them in the recipe for every tile.

    import geozl

    noise = geozl.lossy.fit_noise(stack)    # (N, H, W), or one (H, W)
    print(noise)
    frame = geozl.compress(tile, method="planar>zigzag>entropy",
                           error=noise.recipe(0.5))

There is no encoder class here yet. A frame carrying quant_sqrt is written
through geozl.compress and read back through geozl.decompress.

The fit pools block statistics across everything pushed before it solves, which
is not the same as averaging separate fits. Two tiles that each cover one end of
the intensity range give a usable curve together and neither gives one alone.
"""

from dataclasses import dataclass

import numpy as np

from .._dtype import dtype_code
from .._ffi import _ptr, ffi, lib

# The block side the fit measures on, plus the one-sample border its 3x3 mask
# needs. Below this a raster carries no blocks at all.
_MIN_SIDE = 10


@dataclass(frozen=True)
class Noise:
    """A fitted sigma**2 = a + b*x, with what the fit thought of itself.

    a and b are the curve. The rest says how far to believe it, and the fit
    reports them rather than deciding, because what counts as enough depends on
    how the rasters were chosen.

    blocks  blocks that went into the solve
    bins    intensity bins that held enough blocks to count, out of 24
    range   brightest bin over darkest; near one means the rasters were flat
    colin   mean intensity over its spread; large means a and b are collinear
            and the split between them is arbitrary even when the curve fits
    resid   relative rms of the fit residuals
    """

    a: float
    b: float
    blocks: int
    bins: int
    range: float
    colin: float
    resid: float

    def recipe(self, max_error, *, store=None):
        """A quant_sqrt recipe holding max_error sigmas of this curve.

        The numbers go in at full precision so the recipe resolves to the same
        grid the fit produced, rather than to a nearby one.
        """
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
    """Whatever was handed in, as a list of 2d C-contiguous arrays of one dtype.

    A 3d array is a stack over its first axis. A sequence is taken as it comes,
    so rasters of different shapes pool fine as long as the dtype matches, which
    it has to: the fit reads raw samples through one code.
    """
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
    """Fit sigma**2 = a + b*x over one raster or a stack of them.

    data is a 2d array, a 3d array read as (N, H, W), or any sequence of 2d
    arrays sharing a dtype. Everything goes into one pool and one curve comes
    out. Returns a Noise.

    The method is the scatterplot of local mean against local variance with a low
    quantile per intensity bin, after Abramova and others, SPIE 10004, 2016. The
    same model is Foi, Trimeche, Katkovnik and Egiazarian, IEEE TIP 17(10), 2008.
    A raster smaller than 10 samples on a side carries no blocks and is refused.

    Raises RuntimeError with the reason when the pool does not support a curve,
    which is the honest outcome for rasters that are flat or that cover too
    little of their range. It is not something to work around by fitting per
    tile; measure over more of the product instead.
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
