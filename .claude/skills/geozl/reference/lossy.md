# Bounded-error (near lossless) compression

Three quantizers turn a raster into integer grid indices under a declared per-sample
bound. Sources: `core/src/quant_*/spec.md`, `quant_*_spec.c`, `core/src/lossy/lossy_recipe.c`.

## Contents

1. The three families
2. Recipe grammar and the Python shorthand
3. What the stream stores (`STORE`)
4. How each grid is cut
5. The domain frozen by `graph()`
6. Special values: zero, negatives, NaN, infinity
7. SQRT and the noise model (`fit_noise`)
8. Choosing a family and a budget
9. Verifying the bound

## 1. The three families

| Family | Recipe | Guarantee for every finite sample | CTid | Typical data |
| --- | --- | --- | --- | --- |
| LINEAR | `LINEAR:MAX_ERROR=V` | `abs(x - x^) <= V` | `0x72D781` | elevation, temperature, anything with a tolerance in its own units |
| LOG | `LOG:MAX_ERROR=P%` | `abs(x - x^) <= (P / 100) * abs(x)` for normal (non-subnormal) values | `0x72D782` | values across decades: precipitation, humidity, linear SAR backscatter |
| SQRT | `SQRT:MAX_ERROR=KN[,A=a,B=b]` | `abs(x - x^) <= K * sqrt(a + b*x)` where `a + b*x > 0` | `0x72D783` | photon counting sensors with Poisson-Gaussian noise, optical radiance |

In a recipe graph the quantizer sits after `nodata` and before the predictor.
Lossy frames drop OpenZL's content checksum (the reconstruction is not the input);
the compressed checksum stays.

## 2. Recipe grammar and the Python shorthand

`FAMILY:KEY=VALUE[,KEY=VALUE...]`, keys in any order, numbers with a dot decimal.
Unknown keys, a trailing comma, or a repeated `MAX_ERROR` are refused.

| Family | Keys |
| --- | --- |
| LINEAR | `MAX_ERROR=V` (required, `> 0`), `STORE=INDEX` or `STORE=VALUES` |
| LOG | `MAX_ERROR=P%` (required, `%` required, `0 < P < 100`), `STORE=INDEX` or `STORE=VALUES` (repeats refused) |
| SQRT | `MAX_ERROR=KN` (required, `N` required, `K > 0`), `A=a` (`>= 0`) and `B=b` (`> 0`) together or not at all, `STORE=INDEX` or `STORE=VALUES` |

Python shorthand for `error=` (`graph`, `profile`): a positive number is
`LINEAR:MAX_ERROR=<n>`, `"P%"` is `LOG:MAX_ERROR=P%`, zero in any form is lossless.
SQRT and `STORE` always need the full recipe.

## 3. What the stream stores (`STORE`)

The encoded stream is always integers. `STORE` picks between the grid index `q` and the
reconstructed value. The grid and the bound are the same either way; only the
numbers on the wire and the decode work change.

| Family | Integer input | Float input |
| --- | --- | --- |
| LINEAR | default `INDEX`; `VALUES` allowed | default `INDEX`; `VALUES` needs a whole step (`floor(2V) >= 1`) and an exactly representable top value |
| LOG | always `VALUES`; `STORE=INDEX` is **refused** | default `INDEX`; `VALUES` rounds to whole numbers, refused where that breaks the bound (small magnitudes) |
| SQRT | default `VALUES`; `INDEX` allowed only if the index range fits the element width | default `INDEX`; `VALUES` refused below the magnitude where whole-number rounding holds |

Why it matters: indices are `step` times smaller than values, which bit packers
(`pfor`) and transposed entropy coders exploit. Since 0.14.0 an integer LINEAR frame
stores indices by default; on the project benchmark at `MAX_ERROR=100`,
`planar>zigzag>pfor` moved from 4.08x to 9.37x. Entropy terminals see no change,
because scaling every value by a constant keeps the distribution.

## 4. How each grid is cut

### LINEAR

- Zero-anchored uniform grid, reconstruction `q * step`, clamped to the dtype (and to
  zero when the build raster had no negatives).
- Integers (either `STORE`) and float `VALUES`: `step = floor(2V)`, minimum 1 for
  integers. The real integer bound is `floor(step / 2)`: `error=2.7` gives step 5 and a
  worst error of 2; any `V < 0.5` gives step 1, which is lossless. Integers rebuild in
  integer arithmetic, exact up to `uint64` and `int64`.
- Float `INDEX`: `step = 2 * (V - 2 * eps * maxAbs)`, with `eps` of `2^-11`, `2^-24`,
  `2^-53` for f16, f32, f64. The step depends on the largest magnitude of the build
  raster, so graphs built from different rasters can land on different grids (each
  still within the bound). Refused when the step is not positive (`V` below the type's
  rounding at that magnitude) or the index would overflow the stream.

### LOG

- Levels a constant ratio apart; `step` is the gap in log2 units.
- Float `INDEX`: the grid covers the whole floating-point type, anchored by the type
  (`2^-24`, `2^-149`, `2^-1074` for f16, f32, f64) and the requested bound only. It never
  reads the tile, so cutting a product into different tiles never changes a
  reconstruction. Refused when `P` is finer than the type rebuilds
  (`... at or below what this type rebuilds to ...`) or the index needs more levels
  than the element holds.
- `VALUES`: grid anchored at one with levels rounded to whole numbers; the budget is split
  between the grid and the rounding. On integers, where adjacent levels are closer than
  one, the rounding lands back on the sample, so tight bounds on small integers become
  lossless rather than wrong.

### SQRT

- Substituting `u = sqrt(x + offset)` makes the grid uniform again:
  `c = K * sqrt(b)`, `offset = a / b`, `x^ = (q * step)^2 - offset`. The header carries
  `step` and `offset`; `a` and `b` cannot be recovered from a frame.
- Integers and float `VALUES`: `step = c / 2`, independent of the raster.
- Float `INDEX`: `step` is `c` minus rounding charges measured over the raster range;
  f32 may decode in binary32 arithmetic when that costs at most a quarter of the budget.
- Refused when the raster reaches below `-A/B` (`a shot bound is defined at or above
  -A/B ...`) or needs more levels than the arithmetic or element can resolve.

## 5. The domain frozen by `graph()`

`geozl.graph(raster, method, error=...)` resolves the grid from `raster` and records
its statistics. `compress` then refuses tiles outside that domain, raising
`RuntimeError` with `... build the graph from the full product`:

| Family and mode | A later tile is refused when |
| --- | --- |
| LINEAR and LOG | the build raster had no negative sample and the tile has one (`built without negative samples`) |
| LINEAR, float | the tile's largest magnitude exceeds the build raster's (`built for magnitudes up to ...`) |
| LINEAR, integer | only the negativity rule |
| LOG, float with `STORE=VALUES` | the tile's non-zero magnitudes leave the build range (`built for non-zero magnitudes from ... to ...`) |
| LOG, other modes | only the negativity rule |
| SQRT | any finite value leaves the build raster's `[min, max]` (`built for values from ... to ...`) |

A declared sentinel (`nodata=value`) is ignored by the check in later tiles, **but not
when the plan is resolved**: a sentinel inside the build raster widens LINEAR and LOG
domains, drops the non-negative clamp, and makes SQRT refuse values below `-A/B`. Build
from valid samples (`nodata-and-planes.md` section 3). Non-finite values are ignored by
the check too, and that is the NaN trap in section 6.

Build patterns that satisfy the domain:

```python
# the whole product in memory
g = geozl.graph(product, best, width=tile_cols, planes=1, error=0.5)

# a stack of scenes, compressed later as separate 2-D tiles (recipe carries A and B)
g = geozl.graph(stack, best, width=stack.shape[-1], planes=1, error=recipe)
```

Passing `planes=1` with an explicit `width` keeps the graph geometry equal to the 2-D
tiles you compress, while the domain covers all the data. A SQRT recipe without `A` and
`B` is fitted over rows of `width` across the whole build raster, so these patterns need
`A` and `B` from `fit_noise` (otherwise `N samples do not divide into rows of W`, or a
curve fitted on misaligned rows).

## 6. Special values

- **Zero** reconstructs exactly in LINEAR and LOG (LOG loses the sign of `-0.0`). SQRT has
  no special case for zero: it lands on the nearest level, within the bound but not
  necessarily zero (float32 with `A=11,B=1` rebuilt `0` as `1.2496`; the zero clamp for
  non-negative rasters only removes negative reconstructions).
- **Non-negative data** stays non-negative: when the build raster has no negatives the
  decoder clamps at zero (the SQRT grid is anchored at `-offset` and would otherwise dip
  below zero near the bottom).
- **NaN** is outside every guarantee. Without a `nodata` stage all three families
  reconstruct NaN as `0.0`, silently (verified on float32). NaN handling is decided from
  the build raster: if it contained NaN the graph routes NaN through the mask
  automatically; if not, pass `nodata=float("nan")` explicitly.
- **Infinity** is not refused either, and NaN mode does not cover it: a float32 `+inf`
  came back as a large finite value in every family, the exact value depending on the
  recipe. Replace infinities with a sentinel declared as `nodata=`, or keep them out of
  lossy graphs.
- **Subnormal floats** (LOG): encoded, but no relative bound is guaranteed.

## 7. SQRT and the noise model (`fit_noise`)

A recipe without `A` and `B` is fitted from the raster at `graph()` time, over rows of
`width` across the whole raster (planes are ignored, and `width` must divide the sample
count). That changes the grid per graph, and it fails on data that is not
signal-dependent noise. Fit once over a product instead:

```python
curve = geozl.lossy.fit_noise(stack)   # 2-D array, (N, H, W) stack, or a sequence of 2-D arrays
print(curve)       # sigma^2 = 11.02 + 0.9948*x  [900 blocks, 18/24 bins, range 6.4x, colin 2.2, resid 0.084]
recipe = curve.recipe(1.0)                  # "SQRT:MAX_ERROR=1N,A=...,B=..."
recipe = curve.recipe(1.0, store="VALUES")  # optional STORE
g = geozl.graph(stack, best, width=stack.shape[-1], planes=1, error=recipe)
```

- Every raster must be 2-D, at least 10 x 10, and share one dtype of the 11 supported.
- The estimator measures local variance in 8 x 8 blocks (Immerkaer mask), takes a low
  quantile per intensity bin (24 bins), and fits `variance = a + b*x`, after Abramova et
  al. (2016) and the Poisson-Gaussian model of Foi et al. (2008). Negative intercepts are
  clamped to zero.
- `Noise` fields: `a`, `b`, `blocks` measured, `bins` that held enough blocks (needs at
  least 4 with 12 blocks each), `range` (brightest bin mean over darkest), `colin`
  (larger means `a` and `b` are harder to separate), `resid` (relative RMS residual).
- `fit_noise` raises `RuntimeError` with the reason, for example
  `... not enough to fit a curve ...`, `only N intensity bins held enough blocks ...`,
  `the raster has no dynamic range to fit against`, or
  `the local variance does not grow with the signal, so this is not shot noise`.
- Units: `MAX_ERROR=KN` counts standard deviations of the fitted noise.

## 8. Choosing a family and a budget

1. Decide what "acceptable" means in the data's own terms: metres (LINEAR), percent of
   the value (LOG), or a fraction of sensor noise (SQRT).
2. Profile with the budget, because the best recipe changes when data turns into indices:

   ```python
   rows = geozl.profile(sample, prior=None, error=2.5, reps=3)
   best = rows[0]["graph"]
   g = geozl.graph(product, best, width=cols, planes=1, error=2.5)
   ```
3. Sweep a ladder of budgets per family and plot ratio against RMSE or max error; the
   project notebook compares the three families on one DEM this way. Integer rasters
   under LOG or SQRT tend to deliver about half their stated worst case because
   reconstruction rounds to whole numbers.

## 9. Verifying the bound

```python
def check_linear(tile, frame, v):
    back = geozl.decompress(frame).view(tile.dtype).reshape(tile.shape)
    finite = np.isfinite(tile) if tile.dtype.kind == "f" else np.ones(tile.shape, bool)
    err = np.abs(back.astype(np.float64) - tile.astype(np.float64))[finite]
    assert err.max() <= v, err.max()

def check_log(tile, frame, pct):
    back = geozl.decompress(frame).view(tile.dtype).reshape(tile.shape).astype(np.float64)
    x = tile.astype(np.float64)
    assert (np.abs(back - x) <= (pct / 100.0) * np.abs(x)).all()

def check_sqrt(tile, frame, k, a, b):
    back = geozl.decompress(frame).view(tile.dtype).reshape(tile.shape).astype(np.float64)
    x = tile.astype(np.float64)
    assert (np.abs(back - x) <= k * np.sqrt(a + b * x)).all()
```

Compute errors in float64 (or int64) to avoid wraparound in unsigned subtraction.
