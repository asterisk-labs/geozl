# quant_sqrt

A bound that follows the noise of a photon counting sensor. The slack grows like
the square root of the value, so a reading of 10000 gets ten times the room a
reading of 100 does, and a hundred times what a reading of 1 does.

    SQRT:MAX_ERROR=0.5N                  |x - x^| <= 0.5 * sqrt(a + b*x)
    SQRT:MAX_ERROR=0.5N,A=100,B=1        the same, with the curve given
    SQRT:MAX_ERROR=0.5N,STORE=VALUES     whole numbers in the stream

`MAX_ERROR` counts sigmas of noise and the `N` is required. Without it `0.5`
reads as half a count, which is a plausible typo for half a sigma and would
quantize far finer than intended without ever failing.

`A` and `B` are the noise model, `sigma^2 = a + b*x`. They travel together, and a
recipe carrying neither asks for them to be fitted from the raster instead. One
without the other is refused.

## Why a square root grid

Ask for the gap between levels to be proportional to `sqrt(x + offset)` and the
grid is uniform once you substitute `u = sqrt(x + offset)`. So

    x^ = (q * step)^2 - offset

and since `k*sqrt(a + b*x)` is `k*sqrt(b) * sqrt(x + a/b)`, the whole curve is two
numbers, `step` and `offset`, and those two are what the header carries. `a` and
`b` are recoverable from them and are not stored again.

`sqrt` is the only library call in either direction, and IEEE requires it to be
correctly rounded, so a frame written on one machine rebuilds to the same bits on
another. What that reproducibility does need is `-ffp-contract=off`, since
`t*t - offset` folds into a single FMA on arm64 and would then round once instead
of twice.

## Inputs

One numeric stream of `nbElts` elements, at the element width of the original
type.

## The header

Eighteen bytes, little endian.

    byte 0      element type. 0 to 3 are u8, u16, u32, u64, 4 to 7 the signed
                widths in the same order, then 8 f16, 9 f32, 10 f64
    byte 1      flags
                  bit 0  the encoder saw no negative sample, so the decoder
                         floors at zero
                  bit 1  the stream holds the reconstruction
                  bit 2  the decoder rebuilds in float rather than double
    bytes 2-9   step, an IEEE double
    bytes 10-17 offset, an IEEE double

Corrupt is a type byte outside 0 to 10, a flag bit above bit 2, a `step` whose
square is not a normal double, an `offset` that is not finite or is negative,
bit 1 clear on an integer type, and bit 2 set on anything but float32. One
predicate says all of this, in `quant_sqrt_check.h`, and the encoder, the decoder
and the frame reader all read that one rather than each keeping a list.

The condition on the `step` is on its square, not on the step. The encoder
divides by `step^2`, so a step that is merely positive and finite can still
square to zero, and then the reciprocal is an infinity and every sample folds
onto one level in silence. A negative step squares to a normal number, so the
sign is tested on its own.

Bit 0 is load bearing here in a way it is not for the other curves. Level zero
rebuilds to `-offset`, which is below zero whenever there is a read noise floor,
and on an unsigned type that would wrap.

Bit 2 is not a hint. Both paths are plain IEEE and both are reproducible; the bit
says which arithmetic the step was cut against, so a decoder that took the other
one would hold a different bound than the frame declares.

## The three cases

| input | recipe | stream holds | decode |
| --- | --- | --- | --- |
| 1. integer | any | the reconstruction | copy the block |
| 2. float | `STORE=VALUES` | the reconstruction | one conversion |
| 3. float | default | the index | two multiplies and a subtract |

Case 3, the one that does arithmetic, is nearly as fast as the plain copy. On one
core, a tile of 1M float32 at `-O3` on the x86-64-v2 baseline:

    case 1   integer, memcpy                       28476 MB/s
    case 3   index, float arithmetic               23170 MB/s
    case 2   STORE=VALUES, cast                    13387 MB/s
             index, double arithmetic              10608 MB/s
             index, through a level table           7360 MB/s

The last row is the reason there is no lookup table in the decoder. Caching the
levels and indexing them looks like it should win, and it loses by 3.1x, because
the gather costs more than the two multiplies it replaces and it takes the loop
out of the vector unit.

## Cutting the step

With a step `s` in `u`, the worst error in cell `q` is `s^2 * (q + 0.5)` and the
bound where it happens is `c * s * sqrt(q^2 + q + 0.5)`, whose ratio is under one
for every `q` and approaches it from below. So the grid alone admits `step = c`,
where `c = k * sqrt(b)`, and the offset does not enter.

That holds only if the encoder picks the level nearest the sample **in x**.
Rounding in the warped domain is the obvious thing and costs a third of the step,
because the cell boundaries there are not the midpoints between levels and the
gap is worst at the bottom of cell one, where the ratio reaches 1.5. Taking the
nearest level in x costs the same, one `sqrt` either way, since the midpoint
between levels `q-1` and `q` sits at `(x + offset)/step^2 = q^2 - q + 0.5`.

    q = floor(0.5 + sqrt(t - 0.25))      t = (x + offset) / step^2

On top of the grid sits the rounding when the reconstruction reaches the output
width. What it costs at a sample is `eps * |x| / sqrt(x + offset)`, a fraction of
the sample's own magnitude weighed against the bound where that sample sits. That
grows with `x` above zero, and it also runs away as `x` falls toward `-offset`,
where the bound goes to zero and the rounding does not. So it bites at **whichever
end of the raster is worse** and both are measured. A whole number rounding is a
flat half unit instead, so on an integer it bites at the bottom.

Under that sits the encoder's own arithmetic. The quotient `t` rounds before the
root is taken, which moves the boundary between two levels by a relative
`DBL_EPSILON` or so, and a boundary that moves by a relative amount moves by
`eta * (x + offset)` in `x`. Half the level spacing at `u` is `u * step`, so the
two together fit under `c * u` only if the step gives up `eta * u` first.

    case 3, double arithmetic   step = c - eps * max(g(lo), g(hi)) - eta * u
                                where g(x) = |x| / sqrt(x + offset)
    case 3, float arithmetic    the same, eight loose terms rather than one
    cases 1 and 2               step = c / 2, and a ceiling on the level count

The float arithmetic path is taken when its charge stays under a quarter of the
budget. Measured, in the ordinary range of `c` between 0.01 and 1 with `x` under
1e7, it consumes 5.7e-2 of the bound against 2.4e-2 for the double path. That is
under 0.09 bit per sample given up for 2.2x at decode.

Cases 1 and 2 pay the encoder's arithmetic as a refusal rather than as a charge,
because their step has to stay where the recipe put it. Nothing in `c/2` reads the
raster, and that is deliberate: two tiles of one product have to land on the same
grid, or the same value encodes differently in each and the predictors downstream
read the difference as noise. So the term comes back as a ceiling of
`0.5 / (3 * DBL_EPSILON)` levels over the raster, which is a refusal and not a
parameter. That ceiling is also what catches a step small enough to square to
zero.

The half in cases 1 and 2 is the even split between the grid and the rounding of
the level to a whole number. The grid stops biting below
`sqrt(x + offset) = 1/(2*step)`, and half a unit fits under the remaining `c -
step` exactly there when `step` is `c/2`. It is conservative: measured worst error
on an integer raster sits near half the declared bound, so there is a ratio left
on the table here.

Case 2 is refused rather than served when the raster reaches below
`1/c^2 - offset`, since rounding a small float to a whole number already costs
more than the bound, and when the reconstruction would run past the largest whole
number the float type carries. Never a quiet fallback to case 3.

## Where the bound ends

Below `-a/b`. The model puts the variance at or under zero there and no bound
exists, so a raster reaching below it is refused rather than encoded. Approaching
it from above is not free either, since the bound shrinks toward zero while the
rounding of the reconstruction does not, and that is the `g(lo)` end of the charge
above.

At the top, where `eps * x` overtakes `c * sqrt(x)`, which is near `(c/eps)^2` for
the double path and `(c/8eps)^2` for the float one. Both sit far above anything a
real product holds; a float32 tile at `c = 0.01` would have to reach 4e8.

The bound is read at signed `x` everywhere, including in `quant_sqrt_bound` and
`quant_sqrt_verify`. The variance grows with the signal and not with its
magnitude, so `a + b*x` and never `a + b*|x|`. Folding the curve at zero reports a
bound several times the one the grid was cut against, and every test and fuzzer
that measures a round trip reads that number.

A sample that is not finite has no place on the grid. It encodes as index zero and
the nodata codec in front is what puts it back.

## Fitting the curve

A recipe with no `A` and `B` asks `quant_sqrt_fit` to measure them. The method is
the scatterplot of local mean against local variance with a low quantile per
intensity bin, after Abramova and others, SPIE 10004, 2016. The model itself is
Foi, Trimeche, Katkovnik and Egiazarian, IEEE TIP 17(10) 1737-1754, 2008.

Blocks are 8 by 8 and the high pass is Immerkaer's mask, whose response to any
linear ramp is zero, which is what keeps a smooth gradient from reading as noise.
Each intensity bin contributes its tenth percentile of local variance rather than
its mean, because the mean is unbiased on pure noise and useless on a real scene,
where every textured block in the bin lifts it.

That quantile is biased low and the bias is a constant. The Immerkaer windows
inside a block overlap, so an 8 by 8 block carries 19.3 effective degrees of
freedom rather than 64, and the tenth percentile lands at 0.7904 of the true
sigma. The correction is applied and it belongs to this block size and this mask;
changing either without re-measuring it moves every bound this codec declares.

**The fit reads the raster, so it moves the grid.** Nothing else in this codec
does. Run once per product it is harmless, run once per tile it means the same
value rebuilds differently in different tiles. This file does not enforce which,
because it cannot see how the caller cuts the data. What it reports instead is
how much the fit should be believed.

    bins    intensity bins that held enough blocks
    range   mu_max / mu_min over those bins
    colin   mean intensity over its spread; large means a and b are collinear
    resid   relative rms of the fit residuals

Measured against synthetic rasters with the curve known, a ramp over the full
range recovers sigma within 6% and a smooth fBm scene within 30%. `b` is the
robust half; `a` is the intercept extrapolated past where the data reaches and
moves several fold between seeds. A narrow band, a flat field and a hard textured
scene all produce a plausible curve that means nothing, and only `range` and
`colin` say so. A scene with two levels and nothing between them produces a
negative `b` and is refused outright.

Two rasters that each cover one end of the range give a usable curve together and
neither gives one alone, which is what `quant_sqrt_accum` is for. It pools the
block statistics before anything is fitted, which is not the same as averaging
separate fits. Measured on two halves of one scene, alone they give +8.5% and
+104.9% and pooled they give -7.4%.

Underestimating sigma is safe. It happens on resampled data, where the noise is
correlated and the local variance reads short, and on Sentinel-2 Level-1C, whose
processing chain already suppressed noise at low signal to noise (Uss, Vozel,
Lukin and Chehdi, SPIE 2017, who also show the residual model there is not
univariate). The bound comes out tight, the raster quantizes finer than needed,
and ratio is lost rather than correctness.

Overestimating is the real risk, because the encoder's own check does not catch
it: it verifies against the bound it computed, which is already inflated. The low
quantile and the four numbers above are the only defence.

## Who measures it

`geozl_lossy_resolve` takes a curve and passes it through, and fits one from the
raster in front of it only when the recipe carries no `A` and `B` and the caller
handed over nothing. `geozl_2d_compress_c` hands over nothing, because it sees one
tile and has no way to reach the rest of the product, so a bare `SQRT:MAX_ERROR=VN`
through the 2d entry point is the per-tile case with everything above that says
about it.

The intended path measures once and puts the two numbers in the recipe, which is
what `geozl.lossy.fit_noise` is for on the Python side. It takes an `(N, H, W)` stack or
any sequence of rasters, pools their block statistics through `quant_sqrt_accum`,
solves once, and `Noise.recipe(max_error)` formats the string.

    noise = geozl.lossy.fit_noise(stack)
    g = geozl.graph(stack, method, error=noise.recipe(0.5))
    frame = geozl.compress(tile, graph=g)

The pooling is what makes it worth doing. Over eight 256 by 256 rasters of a
synthetic scene whose curve is `100 + 1.0*x`, the fit returns `51.4 + 1.041*x`.
The same scene cut so one tile holds only the bright end returns `0 + 1.302*x`
with `colin` at 12.5, and pooling that tile with the dark one returns
`72.9 + 1.016*x` with `colin` at 1.1. The bad fit is not detected by the encoder,
only by `colin`, which is why the four numbers come back to the caller rather than
being checked in here.
