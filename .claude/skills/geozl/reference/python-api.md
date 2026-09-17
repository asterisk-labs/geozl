# Python API

Everything here is `geozl` 0.16.0 as implemented in `bindings/python/geozl/_2d.py`,
`_coeffs.py`, `_dtype.py` and `_simd.py`. The examples and error behaviours were run
against a 0.16.0 build.

## Contents

1. Install and runtime
2. `profile`
3. `graph` and `Graph`
4. `compress`
5. `decompress`
6. `coeffs`
7. `register_decoders` and `simd_info`
8. Supported dtypes
9. The `error=` shorthand
10. Patterns: tiling, metadata, parallelism, untrusted input

## 1. Install and runtime

```bash
pip install geozl
```

- Python 3.11 to 3.14. Runtime dependencies: `numpy>=1.24`, `cffi>=1.17`,
  `openzl>=0.2,<0.3`.
- Wheels exist for Linux x86-64 (manylinux 2.28) and macOS arm64. Elsewhere, build
  from a checkout (`make python FULL=ON`, see `contributing.md`).
- Two native libraries ship inside the wheel under `geozl/_lib/`:
  `libgeozl_kernels` (pure C transforms, loaded at import) and `libgeozl` (links
  OpenZL, loaded lazily by the high-level functions).
- Override library discovery with `GEOZL_LIB` (kernels) and `GEOZL_FULL_LIB` (full
  library). `GEOZL_SIMD` caps the vector path (see `debugging.md`).

## 2. `profile`

```text
geozl.profile(tile, *, prior="planar", width=None, planes=None, error=None,
              reps=5, nodata=None, verify=False) -> geozl.ProfileResults
```

Compresses and decompresses every applicable recipe on `tile`, in C, and returns
one row per recipe that succeeded, sorted by `ratio` descending.

| Argument | Meaning |
| --- | --- |
| `prior` | `"planar"` (default), `"med"`, `"delta_w"`, `"delta_n"`, `"average"`, `"wp_static"` or `"delta_1d"`: that predictor plus the `id` branch. `None`: all 8 predictors. `"none"` (or `"id"`): no predictor. Anything else raises `ValueError(... is not one of ...)`. |
| `width`, `planes`, `error`, `nodata` | Same as `graph`. Profile with the settings you will build with. |
| `reps` | Timed round trips per recipe; the fastest rep is reported. `reps < 1` raises `ValueError`. |
| `verify` | Checksum verification during the timed decode only. Frames are always written with checksums, so `bytes` never changes. |

Each row is a `dict`:

| Key | Meaning |
| --- | --- |
| `graph` | Recipe name, valid as `method` for `graph()` |
| `bytes` | Exact size of the frame `compress` writes (checksums included) |
| `ratio` | `tile.nbytes / bytes` |
| `encode_mbps`, `decode_mbps` | Raw MB per second from the fastest rep; when a rep reads as zero time the mean over the run is used, and `inf` only when the whole run does |
| `shannon_pct` | Order-0 byte entropy size of the raw tile over `bytes`, in percent. Above 100 means the graph found structure a byte histogram cannot see |

`ProfileResults` is a `list` subclass with attributes `shape`, `dtype`, `raw_bytes`,
`width`, `planes`, `prior`, `reps`, `error` (normalized recipe or `None`).
`print(rows)` renders them as a header plus a table:

```text
input
  shape    (256, 256)  [rows, columns]
  dtype    uint16  [2 bytes/sample]
  raw      0.13 MB
  profile  1 plane · row width 256 · all predictors · 1 rep · lossless

graph                             ratio  enc MB/s  dec MB/s  shan%
average>zigzag>entropy             6.91     431.2     469.8    569
...
```

Rows whose recipe fails on this tile are skipped silently, which is why a float32
profile with `prior=None` returns 48 rows while the grid lists 56 names (every
plain `>entropy` recipe fails at 4 bytes). Speeds depend on the machine and the
SIMD path; compare rows from one run only.

## 3. `graph` and `Graph`

```text
geozl.graph(raster, method, *, width=None, planes=None, error=None,
            nodata=None) -> geozl.Graph
```

| Argument | Meaning |
| --- | --- |
| `raster` | Array-like, converted with `np.ascontiguousarray`. Native byte order; 1, 2, 4 or 8 bytes per element. |
| `method` | Recipe string, exactly as `profile` spells it. Non-string or empty raises `ValueError`; an unknown name raises `RuntimeError(... unknown method ...)`; a recipe the element width cannot run raises `RuntimeError(... does not apply to N-byte elements ...)`. |
| `width` | Row width in samples. Defaults to `raster.shape[-1]`; required for 1-D input. |
| `planes` | Number of stacked images whose row predictors restart at each boundary. Defaults to `shape[0]` when `ndim >= 3`, else 1. Must split the size into whole rows. |
| `error` | `None` or `0` lossless; number, percentage string or full recipe (section 9, `lossy.md`). |
| `nodata` | `None`: NaN mode only if the raster is floating point **and contains NaN now**. A float NaN: NaN mode. Any other value: sentinel compared by bit pattern at the dtype. See `nodata-and-planes.md`. |

What `graph()` fixes, and what it means later:

- **dtype.** `compress` refuses any other dtype, even at the same width
  (`uint16` graph, `int16` tile raises `ValueError`).
- **Row width and planes.** Encoded into predictor headers. `compress` does not check
  the tile shape against them (trap 4 in `SKILL.md`).
- **Lossy plan.** The quantizer parameters are resolved from this raster, and the
  raster's statistics become the domain `compress` enforces (`lossy.md`).
- **NoData mode.** Decided here, not per tile.
- **Compressor state.** The handle owns an OpenZL compressor and compression context,
  freed by the garbage collector. It is **not thread-safe** and **cannot be pickled**.

`Graph` attributes: `method`, `width`, `planes`, `dtype` (read-only), `itemsize`,
`error` (normalized recipe string or `None`), `nodata` (the argument you passed; it
stays `None` even when NaN mode was detected automatically).

## 4. `compress`

```text
geozl.compress(tile, *, graph, coeffs=None) -> bytes
```

- `graph` must be a `Graph`; passing a recipe string raises `TypeError`.
- `tile.dtype` must equal `graph.dtype` exactly.
- Output is deterministic for the same tile and graph, and every frame decodes on
  its own: frames from one graph do not depend on each other.
- Lossy graphs check the tile against the build domain first and raise
  `RuntimeError(... build the graph from the full product)` when it falls outside.
  A declared sentinel is excluded from that check.
- The output buffer is sized to `1024 + 1.5 * tile.nbytes`, enough for incompressible data.
- `coeffs` attaches integer metadata to this frame only (section 6).

## 5. `decompress`

```text
geozl.decompress(frame, *, verify=True, max_output_size=None) -> np.ndarray
```

- `frame` is any bytes-like object (`bytes`, `bytearray`, `memoryview`, uint8 array).
- Returns a **flat `uint8` array** backed by 8-byte aligned memory. Restore with
  `.view(dtype).reshape(shape)`; the frame does not record either.
- No decoder registration is needed; the C path registers every geozl codec.
- `verify=False` skips both frame checksums, worth roughly 1 to 30 percent of decode
  time. A flipped payload bit may then decode into a different raster without error
  (codecs' own structural checks can still raise).
- `max_output_size` refuses a frame whose declared output exceeds it, before
  allocating (`ValueError`). Use it for untrusted input.
- Errors: `RuntimeError("geozl.decompress: unreadable frame")` when the size cannot
  be read, `RuntimeError("geozl.decompress failed: ...")` for corruption, checksum
  failures or unknown codecs.

## 6. `coeffs`

```python
frame = geozl.compress(tile, graph=g, coeffs=[[7, -3, 11], [2, 5]])
geozl.coeffs(frame)          # ((7, -3, 11), (2, 5)); None when absent
```

- Stored in the OpenZL frame header comment (format version 22 and later), read
  without decompressing the payload. geozl gives the numbers no meaning.
- 1 to 255 non-empty vectors of int32 values; floats, bools and strings raise
  `TypeError`, out-of-range values raise `ValueError`.
- Blob size is `6 + 4 * vectors + 4 * values` and must stay within 10000 bytes
  (at most 2497 values in a single vector).
- Typical uses: scale and offset codes, band ids, tile coordinates.

## 7. `register_decoders` and `simd_info`

- `geozl.register_decoders(dctx)` registers the Python decoders of every codec
  exposed in `geozl.lossless` and `geozl.lossy` on an `openzl.ext.DCtx`. Needed only
  for low-level decoding (`low-level-graphs.md`).
- `geozl.simd_info()` returns `{"built": [...], "cpu": [...], "active": "neon"}`
  with names from `scalar`, `sse2`, `avx2`, `neon`. Frames are identical on every path.

## 8. Supported dtypes

| Use | dtypes |
| --- | --- |
| Lossless | any native dtype of 1, 2, 4 or 8 bytes (`bool`, `complex64` and friends included; the frame only sees bytes) |
| Lossy (`error=`) | `uint8 uint16 uint32 uint64 int8 int16 int32 int64 float16 float32 float64`; others raise `ValueError(the quantizers do not support dtype ...)` |
| Sentinel `nodata=` | the same 11 dtypes, except `float16` (NaN mode still works there) |
| Not accepted | non-native byte order, 16-byte elements such as `complex128` (use the low-level `Deinterleave` path) |

## 9. The `error=` shorthand

| You pass | Becomes |
| --- | --- |
| `None`, `0`, `0.0`, `"0%"`, `"LINEAR:MAX_ERROR=0"`, `"LOG:MAX_ERROR=0%"`, `"SQRT:MAX_ERROR=0N"` | lossless |
| positive `int` or `float` (numpy scalars too) | `"LINEAR:MAX_ERROR=<value>"` |
| `"P%"` with `0 < P < 100` | `"LOG:MAX_ERROR=P%"` |
| any string containing `:` | passed to C unchanged and validated there |
| `bool`, negative, NaN, inf, `"10"`, `"100%"`, `"-1%"` | `ValueError` |

`Graph.error` and `ProfileResults.error` hold the normalized form.

## 10. Patterns

### Tiling a large raster

Only the column count and plane count matter to the predictors; row count is free.
Keep one graph per column count. For lossy graphs build every one from the whole
product so all tiles share one domain and one grid.

```python
import numpy as np
import geozl

def tile_frames(product, method, size, error=None):
    graphs = {}
    out = []
    for r in range(0, product.shape[0], size):
        for c in range(0, product.shape[1], size):
            tile = product[r:r + size, c:c + size]
            cols = tile.shape[1]
            if cols not in graphs:
                graphs[cols] = geozl.graph(product, method, width=cols, planes=1,
                                           error=error)
            out.append(((r, c), tile.shape, geozl.compress(tile, graph=graphs[cols])))
    return out
```

Verified: a 300 x 200 float32 product cut into 128-sample tiles with `error=0.25`
round-trips with a worst error of 0.24998. With a sentinel, build from the valid samples
instead of `product` (`nodata-and-planes.md` section 3); with SQRT, use a recipe that
carries `A` and `B`, because a fit needs `width` to divide the build raster.

### Keeping dtype and shape

The frame will not tell you. Store `dtype.str` and `shape` beside each frame (in the
container index, a Zarr codec config, a manifest, or `coeffs` when a small integer
code is enough).

### Parallelism

`Graph` wraps a C handle: not thread-safe, not picklable. Build a graph per thread,
or inside each worker process from the same recipe and the same build raster. Frames
are independent, so tiles parallelize freely.

### Untrusted frames

```python
out = geozl.decompress(frame, max_output_size=expected_nbytes)
```

Keep `verify=True`. Report malformed frames that crash, hang or exhaust memory as
security issues (email `hello@asterisk.coop`, subject `[geozl security]`).
