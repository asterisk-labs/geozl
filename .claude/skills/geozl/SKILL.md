---
name: geozl
description: >-
  Use GeoZL or work on its codebase: compress numeric raster and Earth-observation
  tiles; select or debug graph recipes, NoData, planes, and bounded-error
  quantization; compose GeoZL codecs in OpenZL graphs; preserve CTid and frame
  compatibility; or edit GeoZL's C/Python codecs, tests, fuzzers, and docs. Do not
  use for generic OpenZL work that does not involve GeoZL.
---

# GeoZL

[OpenZL](https://github.com/facebook/openzl) represents compression as a graph of
small codecs and writes the resolved graph into every frame. A reader can rebuild
that graph when it has every codec used by the frame. GeoZL adds raster-aware
spatial predictors (`planar`, `med`, `wp_static`, ...), a NoData mask codec,
bounded-error quantizers (`quant_linear`, `quant_log`, `quant_sqrt`) and a block
bit packer (`pfor`).

This skill describes **geozl 0.17.0** (OpenZL 0.2.x). Check `geozl.__version__`.
If it differs, trust the installed source and `CHANGELOG.md` over this file.

## Mental model

- A tile enters as one flat numeric stream of 1, 2, 4 or 8 byte elements, and
  `decompress` gives back bytes only: **dtype and shape are yours to store**.
- A recipe string names the pipeline, for example `planar>zigzag>transpose>entropy`.
  The full graph is `[nodata] -> [quantizer] -> [predictor > zigzag] -> terminal`;
  `nodata=` and `error=` add the first two stages, they are never spelled in the string.
- `profile` measures recipes on a sample, `graph` builds one recipe once (and
  freezes lossy parameters), `compress` runs it per tile, `decompress` returns flat `uint8`.

## Canonical workflow

```python
import numpy as np
import geozl

rows = geozl.profile(sample, prior=None, reps=3)  # every recipe, ranked by ratio
print(rows)                                       # input header + table
best = rows[0]["graph"]                           # or pick on the ratio vs decode MB/s front

g = geozl.graph(sample, best)                     # build once per dtype and row width
frames = [geozl.compress(t, graph=g) for t in tiles]

back = geozl.decompress(frames[0]).view(tiles[0].dtype).reshape(tiles[0].shape)
```

Bounded error: pass the **same** `error=` to `profile` and `graph`. `2` means
`|x - x^| <= 2` (LINEAR), `"1%"` means relative (LOG), full recipes allow SQRT and
`STORE=`. `None` and `0` are lossless. Build lossy graphs from data spanning the
whole product, not from the first tile.

## Choosing a graph

Use `profile` on representative data; it benchmarks actual frames but silently skips
recipes that fail. Treat these as starting points, not fixed recommendations.

| Data | Start with |
| --- | --- |
| Smooth 1 or 2 byte rasters (DEM, reflectance) | `planar>zigzag>transpose>entropy` (2 bytes) or `planar>zigzag>entropy` |
| Decode speed matters most | `planar>zigzag>pfor` (fused codec), `delta_1d>pfor` |
| 4 or 8 byte ints and floats | `...>transpose>entropy`, `...>transpose>zstd`, `...>zstd`, `...>field_lz`, `...>pfor`, never plain `>entropy` |
| Masks, classes, land cover (1 or 2 bytes) | `id>categorical` or a predictor plus `categorical` |
| blosc or EOPF style baseline to beat | `id>transpose>zstd` |
| `(B, Y, X)` cube | pass the cube, `planes=B` is inferred; `width=Y*X, planes=1` predicts across bands |
| Absolute tolerance in data units | `error=V` (LINEAR) |
| Values spanning decades | `error="P%"` (LOG) |
| Photon counting sensors, variance `a + b*x` | `SQRT:MAX_ERROR=kN,A=..,B=..` from `geozl.lossy.fit_noise(stack).recipe(k)` |
| Complex SAR (`complex64`, `complex128`) | low-level `Deinterleave` over the component view |

## Invariants and pitfalls

- `decompress` returns flat `uint8`; dtype and shape are external metadata. Restore them
  with `.view(dtype).reshape(shape)` and do not guess either value.
- Recipe spelling is exact. Copy a recipe from `profile`, but remember that it skips
  failures. `transpose>*` needs 2- to 8-byte elements, `categorical` accepts at most
  2-byte elements, and plain `entropy` fails during compression above 2 bytes.
- A `Graph` fixes the exact dtype and predictor geometry, while `compress` does not check
  the ndarray shape. Keep one graph per intended column count and plane layout. A graph
  is neither thread-safe nor picklable.
- Lossy parameters and their accepted domain are resolved from the raster passed to
  `graph`. Build from representative valid data, pass the same `error` to `profile` and
  `graph`, and declare NoData explicitly for product-wide use. NaN and infinity are
  outside the lossy guarantees unless handled before compression.
- Released CTids and wire layouts are compatibility contracts. GeoZL 0.16 planar recipes
  use fused CTids that older readers do not know, so upgrade readers before writers.
  Do not use frame-byte hashes as cross-version content identity.
- Low-level `openzl.ext` compression needs the current format version and must disable
  the content checksum for quantizers. `geozl.register_decoders` registers only codecs
  exposed by the Python packages; the high-level C-backed `geozl.decompress` also knows
  legacy and C-only codecs.
- Keep checksum verification enabled and set `max_output_size` for untrusted frames.
  `blocked_transpose_zstd` is work in progress; do not select it merely because
  it appears in `profile`.

## Reference map

Read only the reference relevant to the current task. Each one identifies its source
files in the repository and is scoped to GeoZL 0.17.0.

| Task | Read |
| --- | --- |
| Python API: arguments, return values, dtypes, tiling, parallelism | [references/python-api.md](references/python-api.md) |
| Recipe grammar, predictors, terminals, width rules, reading `profile` | [references/recipes.md](references/recipes.md) |
| Bounded error: LINEAR, LOG, SQRT, `STORE`, domains, `fit_noise` | [references/lossy.md](references/lossy.md) |
| NoData, NaN, sentinels, planes and cubes | [references/nodata-and-planes.md](references/nodata-and-planes.md) |
| `openzl.ext` graphs with geozl nodes, complex rasters, raw PFOR | [references/low-level-graphs.md](references/low-level-graphs.md) |
| Codec catalog: CTids, streams, header layouts, legacy codecs | [references/codecs.md](references/codecs.md) |
| C API (`geozl.h`), linking, compiled examples | [references/c-api.md](references/c-api.md) |
| Frame compatibility, version history, golden frames, wire changes | [references/compatibility.md](references/compatibility.md) |
| Repository layout, build, tests, fuzzing, adding a codec, CI, release | [references/contributing.md](references/contributing.md) |
| Error message lookup, SIMD paths, library loading | [references/debugging.md](references/debugging.md) |
