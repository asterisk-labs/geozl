---
name: geozl
description: >-
  Expert use of GeoZL (`import geozl`), the raster-aware codec layer for OpenZL,
  and work on its codebase. Use whenever code compresses or decompresses numeric
  rasters, image bands, DEM, SAR or optical tiles, or (bands, rows, cols) cubes
  with geozl or OpenZL: geozl.profile, graph, compress, decompress, coeffs;
  recipe strings chaining planar, med or delta predictors with zigzag, transpose,
  entropy, zstd or pfor; error= bounds (LINEAR, LOG, SQRT, fit_noise); nodata= NaN or sentinels;
  width= or planes=; openzl.ext graphs with geozl.lossless or geozl.lossy nodes;
  CTids 0x72D7xx and frame compatibility; the C API in geozl.h; or editing
  core/src, bindings/python, fuzz, docs or CHANGELOG in the geozl repository
  (new codec, wire format, golden frames). Also use when comparing geozl with
  ZSTD, DEFLATE, LERC or blosc for Earth observation data, or when debugging
  errors such as "unknown method", "does not apply to 1-byte elements" or "the
  lossy graph was built without negative samples".
---

# GeoZL

[OpenZL](https://github.com/facebook/openzl) represents compression as a graph of
small codecs and writes the resolved graph into every frame, so one decoder reads
any graph. GeoZL adds the raster knowledge OpenZL lacks: spatial predictors
(`planar`, `med`, `wp_static`, ...), a NoData mask codec, bounded-error quantizers
(`quant_linear`, `quant_log`, `quant_sqrt`) and a block bit packer (`pfor`).

This skill describes **geozl 0.16.0** (OpenZL 0.2.x). Check `geozl.__version__`.
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

## Choosing

`profile` is the source of truth; these are starting points.

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

## Traps (verified on 0.16.0; do not "fix" by guessing)

1. **`decompress` returns flat `uint8`.** Keep dtype and shape in your container and
   apply `.view(dtype).reshape(shape)`. A wrong view is silent: int16 `-1000` read
   as uint16 is `64536`.
2. **Recipe spelling is exact.** Predictors other than `id` and `delta_1d` need
   `>zigzag>`: `planar>zigzag>pfor`, `id>pfor`, `delta_1d>transpose>zstd`. Copy names
   from `profile` rows. `store_lo` no longer exists.
3. **Terminal vs element width.** `transpose>*` is refused on 1 byte and `categorical`
   above 2 bytes, both at `graph()`. Plain `entropy` on 4 or 8 bytes **builds but fails
   in `compress`** (`Input does not respect conditions for this node`, mentioning
   `eltWidth != 2`). `profile` never raises the C-side recipe, fit or width errors: it
   skips failing rows, so an empty table means "run `geozl.graph` with the same
   arguments to see why".
4. **A graph fixes dtype (exact, including signedness), row width and planes.**
   `compress` does not check shape. A predictor recipe fails with an opaque
   `Input does not respect conditions for this node` when the tile's sample count is not
   a multiple of the graph width (tiles shorter than one row pass as a single row); when
   it is a multiple, the tile is silently predicted with the wrong rows (lossless, worse
   ratio). Keep one graph per column count.
5. **Lossy graphs freeze their plan on the raster given to `graph()`.** Later tiles are
   refused when they hold negatives the build raster lacked (LINEAR, LOG), exceed its
   largest magnitude (float LINEAR), leave its `[min, max]` (SQRT) or its non-zero
   magnitude range (float LOG with `STORE=VALUES`). Build from the product or a stack,
   with explicit `width=` and `planes=` for the tiles you will compress. Sentinels are
   ignored in later tiles but **not in the build raster**: there they widen the domain,
   drop the non-negative clamp, and make SQRT refuse. Build from valid samples only,
   e.g. `geozl.graph(raster[raster != -9999], best, width=cols, planes=1, error=..., nodata=-9999)`.
6. **NaN and infinity.** NaN handling is decided at `graph()` from the build raster.
   Without NaN mode, NaNs in lossy tiles decode as `0.0` silently; pass
   `nodata=float("nan")` for float products (`np.float32("nan")` is not a Python float
   and becomes a one-payload sentinel). NaN mode restores every hole with the first
   payload in the tile, so a **lossless** tile holding several NaN payloads writes a frame
   whose content checksum fails in `decompress`; normalize with `a[np.isnan(a)] = np.nan`.
   `inf` is never a hole: lossy graphs turn it into a large finite value.
7. **Integer LINEAR steps are whole.** `step = floor(2V)`, minimum 1: `error=2.7` gives a
   real bound of 2, and `V < 0.5` is lossless. LOG on integers always stores values;
   `STORE=INDEX` there is refused.
8. **SQRT needs a noise curve.** Without `A=` and `B=` the curve is fitted per graph over
   rows of `width` of the whole build raster (ignoring planes); it fails on data without
   signal-dependent noise ("not shot noise") and when `width` does not divide the raster.
   Fit once with `geozl.lossy.fit_noise(stack)` and pass its `A` and `B`. The `N` suffix is
   required. SQRT rebuilds zero within the bound, not necessarily as exactly zero.
9. **Readers must know every CTid in a frame.** 0.16.0 planar recipes write the fused
   `planar_zigzag` (`0x72D70F`) and `planar_zigzag_pfor` (`0x72D710`) codecs, which
   0.15.x and older cannot decode. Upgrade readers before writers. Frame bytes may change
   between releases, so never use frame hashes as content identity across versions.
10. **`blocked_transpose_zstd` rows** appear in 0.16.0 `profile` output but the codec is
    work in progress, hidden from the Python node API and the docs. Prefer another row.
11. **`Graph` is not thread-safe and cannot be pickled.** Build one per thread or inside
    each worker process.
12. **Low-level `openzl.ext` graphs** need `CParam.FormatVersion = MAX_FORMAT_VERSION`,
    `CParam.ContentChecksum = 2` (disable) when a quantizer is present,
    `geozl.register_decoders(dctx)` before decoding, and A and B in any `QuantSqrt` recipe.
13. **Inputs:** native byte order only; sentinels must fit the dtype exactly
    (`OverflowError`, `ValueError`); `float16` takes NaN but no sentinel; `(T, B, Y, X)`
    infers `planes=T`, so pass `planes=T*B`. Pass `max_output_size=` when decoding
    untrusted frames.

## Reference map

Before writing anything beyond the canonical workflow, open the matching file. Each one
states its sources in the repository, and its examples were run against 0.16.0.

| Task | Read |
| --- | --- |
| Python API: arguments, return values, dtypes, tiling, parallelism | [reference/python-api.md](reference/python-api.md) |
| Recipe grammar, predictors, terminals, width rules, reading `profile` | [reference/recipes.md](reference/recipes.md) |
| Bounded error: LINEAR, LOG, SQRT, `STORE`, domains, `fit_noise` | [reference/lossy.md](reference/lossy.md) |
| NoData, NaN, sentinels, planes and cubes | [reference/nodata-and-planes.md](reference/nodata-and-planes.md) |
| `openzl.ext` graphs with geozl nodes, complex rasters, raw PFOR | [reference/low-level-graphs.md](reference/low-level-graphs.md) |
| Codec catalog: CTids, streams, header layouts, legacy codecs | [reference/codecs.md](reference/codecs.md) |
| C API (`geozl.h`), linking, compiled examples | [reference/c-api.md](reference/c-api.md) |
| Frame compatibility, version history, golden frames, wire changes | [reference/compatibility.md](reference/compatibility.md) |
| Repository layout, build, tests, fuzzing, adding a codec, CI, release | [reference/contributing.md](reference/contributing.md) |
| Error message lookup, SIMD paths, library loading | [reference/debugging.md](reference/debugging.md) |

## Public Python API

| Symbol | Purpose |
| --- | --- |
| `profile(tile, *, prior="planar", width=None, planes=None, error=None, reps=5, nodata=None, verify=False)` | Benchmark candidate recipes, returns list-like `ProfileResults` sorted by ratio |
| `graph(raster, method, *, width=None, planes=None, error=None, nodata=None)` | Build a reusable `Graph` |
| `compress(tile, *, graph, coeffs=None)` | One frame as `bytes` |
| `decompress(frame, *, verify=True, max_output_size=None)` | Flat `uint8` array |
| `coeffs(frame)` | Application int32 vectors from the frame header, or `None` |
| `register_decoders(dctx)` | Teach an `openzl.ext.DCtx` every geozl codec |
| `simd_info()` | Built, CPU and active SIMD paths |
| `geozl.lossless` | `DeltaW DeltaN Planar PlanarZigzag PlanarZigzagPfor Med Average WpStatic Deinterleave Nodata Pfor` (+ `*Decoder`) |
| `geozl.lossy` | `QuantLinear QuantLog QuantSqrt` (+ `*Decoder`), `fit_noise`, `Noise` |

## Working in the geozl repository

```bash
python -m pip install cmake ninja numpy cffi openzl pytest ruff mypy
make submodules            # OpenZL lives in extern/openzl
make python FULL=ON        # build libgeozl + kernels, stage them, editable install
make test                  # C tests, then pytest
ruff check . && mypy
```

A `FULL=OFF` build has no `libgeozl`, so every high-level test skips. Adding a codec
touches C kernels and bindings, both registries, `ctids.h`, `_ffi.py`, a Python module,
a `spec.md` and the docs catalog in five places: follow
[reference/contributing.md](reference/contributing.md). Changing a shipped wire format
follows the checklist in [reference/compatibility.md](reference/compatibility.md).

## Install this skill in another project

```bash
npx skills add asterisk-labs/geozl
```

Useful options: `-g` (user-global), `-a claude-code` or `-a cursor` (one agent),
`-y` (non-interactive). In this repository the skill lives at `.claude/skills/geozl/`.
