# NoData, NaN, sentinels, planes and cubes

Sources: `core/src/nodata/` (codec and `spec.md`), `bindings/python/geozl/_2d.py`
(`_nodata_args`, `_prepare`), `_dtype.py` (`nodata_bits`), `core/src/common/raster.h`.

## Contents

1. The `nodata=` argument
2. What the nodata codec does
3. NoData with bounded error
4. Low-level `Nodata` node
5. Planes: stacked images
6. Geometry rules and 4-D arrays

## 1. The `nodata=` argument

`geozl.graph(..., nodata=)` and `geozl.profile(..., nodata=)` decide the mode once,
from the raster passed in:

| You pass | Mode | Notes |
| --- | --- | --- |
| `None`, float raster containing NaN now | NaN | every NaN becomes a hole |
| `None`, anything else | none | no nodata stage; a later NaN in a lossless graph still round-trips bit for bit, in a lossy graph it decodes as `0.0` |
| `float("nan")`, `np.nan`, `np.float64("nan")` | NaN | Python `float` NaN, independent of the raster |
| `np.float32("nan")` | **sentinel** | not a Python `float`, so it matches only that exact NaN bit pattern; other NaN payloads are not holes |
| a number | sentinel | compared by bit pattern at the raster dtype |

Sentinel rules (`nodata_bits`):

- It must be a whole number on integer dtypes (`nodata=3.5` on int raises `ValueError`).
- It must fit the dtype (`nodata=-9999` on `uint16` raises `OverflowError`).
- `float16` has no sentinel support; NaN mode still works.
- The dtype must be one of the 11 geozl dtypes (not `bool`, not complex).
- It crosses into C as bits, so `int64` and `uint64` sentinels above `2^53` work.
- Floats compare bits: `-0.0` and `0.0` are different sentinels.

Declaring a sentinel that a tile does not contain costs almost nothing: the mask is all
valid and codes to a few bytes. For lossless graphs declaring it for a whole product is
safe; for lossy graphs keep the sentinel out of the build raster (section 3).

## 2. What the nodata codec does

CTid `0x72D70C`. One numeric stream in, two out:

- `values`: the raster with every hole filled, sent down the rest of the recipe.
- `mask`: one byte per sample, `0` for NoData, nonzero for valid, sent to OpenZL generic
  compression.
- Header: the NoData bit pattern at the element width (plain form), or the pattern
  followed by its neighbour above, below and the other signed zero (guarded form,
  `4 * eltWidth` bytes). The length tells a reader which form it holds.

Fill rule, row by row: a hole takes the last valid sample to its left in the same row;
a hole that opens a row takes the (already filled) sample above; a hole that opens the
first row takes zero. Predictors therefore see a continuous surface instead of a cliff.

Decode: `out[i] = mask[i] == 0 ? pattern : values[i]`.

**Guarded form.** Sentinels use mask codes 1 (above), 2 (below) and 3 (the other
signed zero). When a quantized value matches the sentinel, the decoder uses the
adjacent value named by the mask.

The graph builder records sides only within the quantizer's reach: 0 for lossless,
`MAX_ERROR` for LINEAR, `p|S|/(1-p)` for LOG, and the corresponding SQRT root. The
low-level node records every side. NaN uses the plain form.

NaN mode masks every NaN whatever its payload, but the header stores one pattern (the
first NaN found), so all holes decode with that payload. On a **lossless** graph a tile
with several payloads therefore writes a frame whose content checksum no longer matches:
`geozl.decompress` raises `Content checksum mismatch` (only `verify=False` reads it), and
`profile` does not notice because it times decode without verification. Normalize
payloads first (`a[np.isnan(a)] = np.nan`), or, when payloads carry meaning, compress
losslessly without NaN mode (pass a sentinel the data never holds, which disables the
automatic NaN detection).

`inf` and `-inf` are values, not holes, in every mode.

The codec refuses an empty tile. It runs before the quantizer and the predictor.

## 3. NoData with bounded error

- In **later tiles** the sentinel is excluded from the lossy domain check.
- In the **build raster** it is not. `graph()` resolves the quantizer from the raw raster,
  sentinel included: a `-9999` sentinel widens LINEAR and LOG domains, removes the
  non-negative clamp (so tiles with real negatives are accepted), and makes SQRT refuse
  (`a shot bound is defined at or above -A/B ...`). Build from the valid samples, which
  keeps holes exact and the domain honest (verified for all three families):

  ```python
  valid = raster[raster != -9999.0]      # 1-D, sentinel free
  g = geozl.graph(valid, best, width=raster.shape[1], planes=1, error=0.5, nodata=-9999.0)
  frame = geozl.compress(raster, graph=g)
  ```

  A SQRT recipe fitted without `A` and `B` cannot use this (the fit needs whole rows);
  give it `A` and `B` from `fit_noise`.
- Hole positions decode exactly to the sentinel or NaN; the bound applies to valid samples.
- Make NaN handling explicit on float products:

  ```python
  g = geozl.graph(product, best, error=0.5, nodata=float("nan"))
  ```

  Without it, a product whose build raster has no NaN will turn later NaNs into `0.0`.
- Filled values stay inside the grid because the fill copies valid neighbours.

## 4. Low-level `Nodata` node

```python
import numpy as np
import openzl.ext as zl
import geozl

W = tile.shape[1]
c = zl.Compressor()
values = geozl.lossless.PlanarZigzag(W)(c, zl.graphs.Compress()(c))
start = geozl.lossless.Nodata(W, value=-9999.0, dtype=np.float32)(
    c, values, zl.graphs.Compress())          # (compressor, values_successor, mask_successor)
c.select_starting_graph(start)
```

- `Nodata(width, value=None, dtype=None)`: `value=None` is NaN mode; a sentinel needs
  `dtype` because its bit pattern depends on the type, and because the guard reads the
  samples at that type. A stream of another width is refused.
- Sentinels use the guarded form from section 2. `geozl.graph` supplies its radius;
  the low-level node records every side.
- The two successors may be `GraphID`s or unbound graphs such as `zl.graphs.Compress()`.
- Omit the node for data that never has holes.

## 5. Planes: stacked images

A `(B, Y, X)` cube passed to `graph` or `profile` is `B` planes of rows `X` wide. Row
predictors restart at each plane boundary, so nothing is predicted across bands:

| Codec | Planes |
| --- | --- |
| `planar`, `planar_zigzag`, `planar_zigzag_pfor`, `delta_n`, `average`, `med`, `wp_static` | restart per plane (`wp_static` shares one set of weights across planes) |
| `delta_w` | reads only to its left, takes no plane count (`DeltaW(width, planes=2)` raises `ValueError`) |
| `delta_1d`, `id`, terminals | see a flat stream |

Headers carry the plane count only when it is above one, so single-plane frames look
exactly as before 0.12.0.

Options for multispectral data:

```python
B, Y, X = cube.shape
g = geozl.graph(cube, best)                          # planes=B inferred: bands independent, one frame
g = geozl.graph(cube, best, width=Y * X, planes=1)   # each band is one row: N is the same pixel in the previous band

band_graph = geozl.graph(cube[0], best)              # one frame per band, random access by band
frames = [geozl.compress(band, graph=band_graph) for band in cube]
```

The inter-band layout helps when bands are strongly correlated sample for sample;
it hurts when bands differ in scale. Profile both layouts:

```python
a = geozl.profile(cube, prior=None, reps=3)
b = geozl.profile(cube, prior=None, reps=3, width=cube.shape[1] * cube.shape[2], planes=1)
```

## 6. Geometry rules and 4-D arrays

- `width` defaults to the last axis; 1-D input needs `width=`.
- `planes` defaults to the first axis when `ndim >= 3`. A `(T, B, Y, X)` array therefore
  gets `planes=T` and treats each band boundary inside a time step as an ordinary row
  boundary. Pass `planes=T * B` to restart per band.
- In the high-level API, `planes > 1` must split the size into whole rows, otherwise
  `ValueError`. Low-level C nodes quietly treat an invalid plane layout as one plane.
- A row width of 0 or wider than the stream is stored as one row covering the whole
  stream (low-level nodes).
- `compress` never compares the tile shape with the graph. Only the column count and the
  plane count matter, so tiles with fewer rows reuse the same graph; tiles with a
  different column count or band count need their own graph.
