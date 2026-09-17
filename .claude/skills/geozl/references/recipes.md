# Recipes: grammar, predictors, terminals

The recipe grammar lives in `core/src/2d/2d.c` (`candidate_name`, `build_candidate`,
`geozl_2d_grid_c`). Python only forwards the string.

## Contents

1. Grammar and the full pipeline
2. Predictors
3. Terminals
4. Which recipes run at which element width
5. Fused codecs behind recipe names
6. Reading a profile and picking a row
7. Heuristics

## 1. Grammar and the full pipeline

```text
<predictor>>zigzag><terminal>     predictor in delta_w, delta_n, planar, med, average, wp_static
delta_1d><terminal>               OpenZL integer delta, no zigzag
id><terminal>                     no predictor
```

`<terminal>` is one of `entropy`, `field_lz`, `zstd`, `categorical`,
`transpose>entropy`, `transpose>zstd`, `blocked_transpose_zstd`, `pfor`.

That gives 8 predictors x 8 terminals = 64 names. Anything else, such as
`planar>entropy`, `id>zigzag>entropy`, `planar_zigzag>entropy` or the removed
`store_lo`, is `unknown method`.

The graph `geozl.graph` builds around a recipe:

```text
[nodata] -> [quantizer] -> [predictor -> zigzag] -> terminal
   |
   +-> validity mask -> OpenZL generic compression
```

- `nodata` appears only when `nodata=` is set or NaN is detected.
- The quantizer appears only when `error=` is lossy. It runs **before** the predictor,
  so predictors and packers see small integer indices, not floats.
- Zigzag folds signed residuals to unsigned (`0, -1, 1, -2 -> 0, 1, 2, 3`).

## 2. Predictors

`W`, `N`, `NW`, `NE`, `NN` are reconstructed neighbours; missing neighbours are zero.
All arithmetic wraps at the element width, so floats and signed integers round-trip
bit for bit through their integer representation.

| Name | Prediction | Planes | Good for |
| --- | --- | --- | --- |
| `planar` | `W + N - NW` | yes | smooth 2-D gradients; a perfect plane becomes zeros |
| `delta_w` | `W` | no (reads left only) | fields smooth west to east |
| `delta_n` | `N` | yes | fields smooth north to south |
| `med` | JPEG-LS median edge detector: `min(W,N)` if `NW >= max(W,N)`, `max(W,N)` if `NW <= min(W,N)`, else `W + N - NW` | yes | rasters with sharp edges |
| `average` | `floor((W + N) / 2)` computed without overflow | yes | surfaces varying in both directions without strong edges |
| `wp_static` | `W + ((cN*N + cNW*NW + cNE*NE + cNN*NN + round) >> shift)`, four int16 weights and a shift fitted once per tile and stored in the header | yes (one weight set for all planes) | when a learned fixed predictor beats the simple ones; the trainer keeps the planar weights `(1, -1, 0, 0)` unless a fit lowers its residual entropy estimate |
| `delta_1d` | OpenZL `DELTA_INT` over the flat stream | n/a | 1-D-like data, fast decode |
| `id` | none | n/a | baseline, noise, data already decorrelated |

`profile(prior=...)` restricts the sweep to one predictor plus `id`.

## 3. Terminals

| Terminal | What it does | Element widths | Notes |
| --- | --- | --- | --- |
| `entropy` | OpenZL adaptive FSE or Huffman on the numeric stream | 1, 2 only in practice | builds at 4 and 8 bytes but `compress` fails (`Input does not respect conditions for this node`, mentioning `eltWidth != 2`) |
| `field_lz` | OpenZL field-aware LZ on the numeric stream | 1, 2, 4, 8 | catches repetition entropy misses |
| `zstd` | numeric to serial little endian, then Zstandard | 1, 2, 4, 8 | |
| `categorical` | one pass counts the dominant value's share: all equal goes to `constant`, above 0.95 to `field_lz`, else `entropy`; the frame records the arm | 1, 2 (refused above at build) | cloud masks sit near 0.97, land cover near 0.50 |
| `transpose>entropy` | split elements into byte lanes (low byte first), entropy per lane | 2, 4, 8 (refused at 1) | the byte shuffle of blosc and EOPF |
| `transpose>zstd` | byte lanes, Zstandard per lane | 2, 4, 8 (refused at 1) | `id>transpose>zstd` is the EOPF-style baseline |
| `blocked_transpose_zstd` | fused shuffle plus independent Zstandard frames per 2 MiB block | 1, 2, 4, 8 | **work in progress** in 0.16.0, not in the docs or Python node API |
| `pfor` | blocks of 256 packed at the bit width with the smallest estimated size, overflowing values patched as exceptions | 1, 2, 4, 8 | very fast decode; cost grows with residual magnitude, so it pairs well with integer quantizer indices |

## 4. Which recipes run at which element width

| Element width | Names in the grid | Usable in `compress` | Excluded |
| --- | --- | --- | --- |
| 1 byte (`uint8`, `int8`, `bool`) | 48 | 48 | `transpose>entropy`, `transpose>zstd` |
| 2 bytes (`uint16`, `int16`, `float16`) | 64 | 64 | none |
| 4 bytes (`uint32`, `int32`, `float32`) | 56 | 48 | `categorical` at build; plain `entropy` fails at compress |
| 8 bytes (`uint64`, `int64`, `float64`) | 56 | 48 | same as 4 bytes |

With the default `prior="planar"` the sweep has 12, 16, 14 (12 usable) and 14
(12 usable) names respectively.

## 5. Fused codecs behind recipe names

Since 0.16.0 the planar recipes do not build separate `planar` and `zigzag` nodes:

| Recipe | Codec actually written |
| --- | --- |
| `planar>zigzag>pfor` | `planar_zigzag_pfor` (CTid `0x72D710`) into `store`; payload byte-identical to `planar_zigzag` then `pfor` |
| any other `planar>zigzag>...` | `planar_zigzag` (CTid `0x72D70F`) then the terminal |
| every other predictor | predictor node, then OpenZL `zigzag`, then the terminal |

Old frames written with separate `planar`, `zigzag` and `pfor` nodes still decode.
New frames need a 0.16.0 or later reader (`compatibility.md`).

## 6. Reading a profile and picking a row

```python
rows = geozl.profile(sample, prior=None, reps=5)
best_ratio = rows[0]["graph"]

def front(rows, speed="decode_mbps"):
    """Rows nothing else beats on both ratio and speed."""
    keep, fastest = [], 0.0
    for row in rows:                      # already sorted by ratio
        if row[speed] > fastest:
            keep.append(row)
            fastest = row[speed]
    return keep

for row in front(rows):
    print(f"{row['graph']:36} {row['ratio']:6.2f}x {row['decode_mbps']:8.0f} MB/s")
```

- Profile a representative tile: same dtype, same row width, same NoData and the
  same `error` you will build with. `bytes` is exact for that tile.
- Ranking is by ratio; the smallest frame is often not the fastest decode.
- Use `reps` of 3 to 5 and compare speeds within one run on one machine.
- Any row's `graph` string is valid for `geozl.graph`.
- Reference point from the project docs: a 512 x 512 int16 Copernicus GLO-30 tile
  (Mont Blanc) reaches 4.50x lossless with `planar>zigzag>transpose>entropy`, and
  9.93x with `error=4`.

## 7. Heuristics

These explain rankings; they do not replace `profile`.

- Continuous physical fields (elevation, temperature, reflectance) reward
  `planar`, `med`, `average` or `wp_static`. Noisy data narrows the gap to `id`.
- After a quantizer, integer indices are small, so `pfor` and transpose terminals gain
  a lot (on the project benchmark `planar>zigzag>pfor` went from 4.08x to 9.37x at
  `MAX_ERROR=100` when integer frames started storing indices).
- Categorical rasters (masks, classes) prefer `categorical`, `entropy` or `field_lz`
  over spatial prediction.
- Multi-byte floats without quantization rarely have repeated byte patterns; lane
  transposition helps because sign and exponent bytes are predictable.
- LZ terminals (`zstd`, `field_lz`) win when exact runs repeat (constant regions,
  filled NoData).
