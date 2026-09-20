# Debugging

Messages below were captured from geozl 0.18.0 with OpenZL 0.2. Numeric
`ZL error code` values come from OpenZL and may move; match on the text.

## Contents

1. Error lookup
2. Unknown codec IDs in a frame
3. Library loading
4. SIMD paths
5. Checking results
6. Reproducing and reporting crashes

## 1. Error lookup

### Building a graph

`geozl.profile` raises the Python-side `ValueError` and `OverflowError` rows below, but
never the `RuntimeError` rows that come from C (full recipe grammar, SQRT fit, grid
limits, element width): it skips every recipe that fails, prints `no compatible graphs`
when none survive, and returns an empty list. Call `geozl.graph` with the same arguments
to see the reason.

| Message (excerpt) | Cause | Fix |
| --- | --- | --- |
| `ValueError: dtype >u2 is not native byte order` | big-endian array | `arr.astype(arr.dtype.newbyteorder("="))` |
| `ValueError: dtype complex128 is 16 bytes per element` | element width not 1, 2, 4 or 8 | view as components and use the low-level `Deinterleave` path |
| `ValueError: give width for a 1d raster` | 1-D input | pass `width=` |
| `ValueError: N planes do not split M elements into whole rows of W` | bad `planes` or `width` | fix the geometry |
| `ValueError: method must be a recipe name` | `method` is not a non-empty `str` | pass a recipe string |
| `RuntimeError: geozl.graph failed (...): unknown method "planar>entropy"` | misspelled recipe | copy from `profile` rows; predictors need `>zigzag>`; `id` and `delta_1d` must not have it; `store_lo` is gone |
| `RuntimeError: ... does not apply to 1-byte elements; the transpose terminals need 2 to 8, categorical needs 1 or 2` | terminal vs element width | choose another terminal (`recipes.md` section 4) |
| `ValueError: prior 'x' is not one of (...) or None` | bad `prior` | `planar med delta_w delta_n average wp_static delta_1d none` or `None` |
| `ValueError: reps must be at least 1` | `reps=0` | use `reps >= 1` |
| `ValueError: the quantizers do not support dtype bool` | lossy on an unsupported dtype | cast to one of the 11 geozl dtypes |
| `ValueError: error must be a number, a percentage such as "10%", or a LINEAR, LOG or SQRT recipe, got '10'` | numeric string | pass `10` or `"LINEAR:MAX_ERROR=10"` |
| `ValueError: relative error must be between 0% and 100%` | `"100%"` or more | use a smaller percentage |
| `RuntimeError: ... MAX_ERROR takes one number ending in N` | SQRT recipe without `N` | `SQRT:MAX_ERROR=0.5N` |
| `RuntimeError: ... A and B travel together` | only one of `A`, `B` | give both, or neither to fit |
| `RuntimeError: ... the local variance does not grow with the signal, so this is not shot noise` | SQRT fit on data without signal-dependent noise | use LINEAR or LOG, or give `A` and `B` |
| `RuntimeError: ... a shot bound is defined at or above -A/B, which is -4, and this raster reaches -10` | data below the SQRT model's domain | shift data or choose another family |
| `RuntimeError: ... STORE=INDEX is not available for integer input on this family` | LOG with `STORE=INDEX` on integers | drop `STORE=INDEX` |
| `RuntimeError: ... is at or below what this type rebuilds to, which bottoms out near ...%` | LOG bound finer than float precision | loosen the bound or go lossless |
| `RuntimeError: ... STORE=VALUES needs a whole step, and a MAX_ERROR of 0.4 gives 0.8` | float LINEAR `VALUES` with `V < 0.5` | use the default `INDEX` or a larger `V` |
| `OverflowError: Python integer -9999 out of bounds for uint16` | sentinel outside the dtype | choose a representable sentinel |
| `ValueError: nodata 3.5 is not a whole number, it cannot be a sentinel on int16` | fractional sentinel on integers | use an integer sentinel |
| `ValueError: a half float carries no sentinel, NaN still works on one` | sentinel on `float16` | use NaN, or store as float32 |
| `ValueError: nodata needs a dtype geozl knows` | sentinel on `bool` and similar | cast first |

### Compressing

| Message (excerpt) | Cause | Fix |
| --- | --- | --- |
| `TypeError: graph must be a geozl.Graph, got str` | recipe passed to `compress` | `compress(tile, graph=geozl.graph(sample, recipe))` |
| `ValueError: this graph was built for dtype int16, got uint16` | dtype differs, even only in sign | build a graph per dtype or cast |
| `RuntimeError: geozl.compress failed (...): Code: Input does not respect conditions for this node` with `eltWidth != 2` in the message (ZL error code 55) | plain `entropy` terminal on 4 or 8 byte elements | pick a transpose, zstd, field_lz or pfor recipe |
| `RuntimeError: geozl.compress failed (...): Code: Input does not respect conditions for this node ... Transform ID: ...` (ZL error code 55) with a predictor recipe | tile sample count not a multiple of the graph's row width | build a graph with this tile's column count |
| `RuntimeError: ... the lossy graph was built without negative samples, but this tile contains them; build the graph from the full product` | domain frozen at `graph()` | build from data spanning the product |
| `RuntimeError: ... built for magnitudes up to A, but this tile reaches B` | float LINEAR domain | same |
| `RuntimeError: ... built for values from lo to hi, but this tile reaches ...` | SQRT domain | same |
| `RuntimeError: ... built for non-zero magnitudes from ... to ...` | float LOG `STORE=VALUES` domain | same, or use `INDEX` |
| `TypeError: coefficient is not an integer: 1.5` | float in `coeffs` | integers only |
| `ValueError: coeffs exceeds the 10000-byte limit` | too much metadata | fewer or shorter vectors |

Silent outcomes with no error, worth checking when results look wrong:

- NaN turned into `0.0`: lossy graph without NaN mode (pass `nodata=float("nan")`).
- `inf` turned into a large finite number: lossy graphs never treat infinity as a hole,
  NaN mode included; replace it with a declared sentinel.
- Lossy tiles with real negatives or out-of-range values accepted: the build raster
  contained the sentinel, which widened the domain; build from valid samples only.
- Poor ratio on edge tiles: the sample count was a multiple of a different row width, so
  the tile was predicted with the wrong rows. Tiles narrower than one graph row, and all
  `id` and `delta_1d` recipes, never trigger the width error at all.
- A `-1000` became `64536`: `.view` with the wrong dtype after `decompress`.
- A different NaN payload than the input after `verify=False`: NaN mode restores every
  hole with the first payload found (on lossless graphs this also breaks the checksum,
  see below).

### Decompressing

| Message (excerpt) | Cause | Fix |
| --- | --- | --- |
| `RuntimeError: geozl.decompress: unreadable frame` | not an OpenZL frame, or the size field is unreadable | check the bytes and offsets you stored |
| `RuntimeError: geozl.decompress failed: Code: Source size too small` | truncated frame | fix storage or range reads |
| `RuntimeError: geozl.decompress failed: Code: Content checksum mismatch` on a frame you just wrote | lossless graph in NaN mode over a tile with more than one NaN payload: every hole decodes with the first payload, so the content hash differs (`profile` still lists the recipe because it times decode with `verify=False`) | normalize payloads before compressing: `a[np.isnan(a)] = np.nan` |
| `RuntimeError: geozl.decompress failed: ...` on flipped bytes | checksum or corruption detection | the frame is damaged; `verify=False` will not repair it |
| `... Custom decoder transform 7526160 not found!` | the reader lacks that codec | section 2 |
| `ValueError: geozl.decompress: the frame declares N bytes of output, above the M allowed` | `max_output_size` guard | raise the limit only for trusted frames |

### Low-level `openzl.ext`

| Symptom | Cause |
| --- | --- |
| `RuntimeError: OpenZL error code: 61` on `compress` | `CParam.FormatVersion` not set |
| `RuntimeError: OpenZL error code: 1` with `disable ContentChecksum on the CCtx` in the traceback | quantizer under the default content checksum |
| `RuntimeError: OpenZL error code: 1` from `QuantSqrt` | recipe without `A` and `B` |
| `RuntimeError: OpenZL error code: 30`, `Custom decoder transform ... not found!` | `geozl.register_decoders(dctx)` missing, or a codec without a Python decoder (legacy pcodec family, `blocked_transpose_zstd`); decode with `geozl.decompress` instead |
| `ValueError: geozl.lossless.delta_w does not support planes` | `DeltaW(width, planes=...)` |

## 2. Unknown codec IDs in a frame

OpenZL prints CTids in decimal. Convert with `hex(7526160)` giving `0x72d710`, then look
it up in `codecs.md`:

- A geozl CTid the reader lacks means the reader is older than the writer (for example a
  0.15.x reader and a 0.16.0 `planar>zigzag>pfor` frame). Upgrade the reader.
- `0x72D780` means a 0.7.x lossy frame; no current reader decodes it.
- An id outside `0x72D700` to `0x72D7FF` is not a geozl codec; the frame needs that
  other library's decoders.

## 3. Library loading

| Message | Meaning |
| --- | --- |
| `OSError: libgeozl_kernels not found, set GEOZL_LIB to its path` | source checkout without staged libraries: run `make python FULL=ON` or `make lib` |
| `OSError: libgeozl not found, set GEOZL_FULL_LIB to its path` | `FULL=OFF` build or missing full library; low-level nodes work, `geozl.graph` and friends do not |
| tests skipped with `libgeozl not built, rebuild with FULL=ON` | same cause, in pytest |

Lookup order: environment variable, then `geozl/_lib/` inside the package, then the
system library path. After changing C code, re-stage the libraries before testing.

## 4. SIMD paths

```python
geozl.simd_info()   # {'built': ['scalar', 'neon'], 'cpu': ['scalar', 'neon'], 'active': 'neon'}
```

- `built`: paths compiled into this binary; `cpu`: paths this machine supports; `active`:
  the one kernels use.
- `GEOZL_SIMD=scalar|sse2|avx2|neon`, set before the library loads (a fresh process),
  caps the active path; unknown values are ignored.
- A C build with `-DGEOZL_NO_SIMD` carries only `scalar`.
- Frames and reconstructions are identical on every path; only speed differs. A wheel
  whose `cpu` list has a path missing from `built` lost a fast path at build time.

## 5. Checking results

```python
import numpy as np

def assert_lossless(tile, frame):
    back = geozl.decompress(frame).view(tile.dtype).reshape(tile.shape)
    # compare bits so NaN payloads and -0.0 count
    bits = np.dtype(f"u{tile.dtype.itemsize}")
    assert np.array_equal(back.view(bits), tile.view(bits))

def frame_matches_profile(tile, row, **graph_kwargs):
    frame = geozl.compress(tile, graph=geozl.graph(tile, row["graph"], **graph_kwargs))
    return len(frame) == row["bytes"]      # True: profile reports the exact frame size
```

For bounded error, see the checks in `lossy.md` section 9.

## 6. Reproducing and reporting crashes

- Inside the repository: `make test-san` (ASan and UBSan), `make fuzz-replay` for saved
  inputs, `make fuzz FUZZ_TIME=600` for a campaign; findings land in `fuzz/out/`.
- Save the offending frame bytes; for decode crashes that is the whole reproducer.
- Crashes, hangs or unbounded allocations on malformed frames are security issues: email
  `hello@asterisk.coop` with `[geozl security]`, not a public issue.
