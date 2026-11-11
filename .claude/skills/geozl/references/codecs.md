# Codec catalog and wire layouts

The CTid is a codec's only identity on the wire; names never travel. Every CTid is in
`core/include/geozl/ctids.h`; each codec folder under `core/src/<name>/` has a
`spec.md` with the full decoder contract. All header integers are little endian.

## Contents

1. CTid bands and helpers
2. Catalog
3. Header layouts
4. Decoding formulas
5. dtype codes
6. Retired CTids

## 1. CTid bands and helpers

| Range | Meaning |
| --- | --- |
| `0x72D700` to `0x72D77F` | lossless codecs |
| `0x72D780` to `0x72D7FF` | lossy codecs |

`geozl_owns_ctid(ctid)` tests the whole range and `geozl_ctid_is_lossy(ctid)` reads the
kind off the band (C API). A codec placed in the wrong band would misreport whether it
destroys data. Released CTids are never reassigned.

## 2. Catalog

| Codec | CTid | Kind | Streams | Python node | Summary |
| --- | ---: | --- | --- | --- | --- |
| `delta_w` | `0x72D701` | predictor | numeric -> numeric | `DeltaW(width)` | residual against the west neighbour |
| `delta_n` | `0x72D702` | predictor | numeric -> numeric | `DeltaN(width, planes)` | residual against the north neighbour |
| `planar` | `0x72D703` | predictor | numeric -> numeric | `Planar(width, planes)` | `W + N - NW` |
| `deinterleave` | `0x72D704` | split | numeric -> numeric x 2 | `Deinterleave()` | even and odd positions into two lanes |
| `med` | `0x72D705` | predictor | numeric -> numeric | `Med(width, planes)` | median edge detector (JPEG-LS) |
| `average` | `0x72D706` | predictor | numeric -> numeric | `Average(width, planes)` | `floor((W + N) / 2)` |
| `wp_static` | `0x72D707` | predictor | numeric -> numeric | `WpStatic(width, planes)` | fixed-weight JPEG XL self-correcting predictor, weights in header |
| `nodata` | `0x72D70C` | split | numeric -> values + mask | `Nodata(width, value, dtype)` | holes into a validity mask, filled values onward |
| `pfor` | `0x72D70D` | terminal | numeric -> serial | `Pfor()` | 256-value blocks bit packed with patched exceptions |
| `planar_zigzag` | `0x72D70F` | fused | numeric -> numeric | `PlanarZigzag(width, planes)` | planar then Zigzag in one pass |
| `planar_zigzag_pfor` | `0x72D710` | fused | numeric -> serial | `PlanarZigzagPfor(width, planes)` | planar, Zigzag and PFOR in 16-row bands cut into strips, 256-value 2D blocks with their headers first |
| `med_zigzag` | `0x72D711` | fused | numeric -> numeric | `MedZigzag(width, planes)` | MED then Zigzag in one pass |
| `planar_zigzag_lanes` | `0x72D712` | fused | numeric -> weights + serial | none, C only | planar and Zigzag, then PivCo Huffman or byte PFOR per residual byte lane |
| `lanes` | `0x72D713` | terminal | numeric -> weights + serial | none, C only | byte lanes, each encoded with PivCo Huffman or byte PFOR |
| `quant_linear` | `0x72D781` | lossy | numeric -> numeric | `QuantLinear(recipe, dtype)` | uniform grid, absolute bound |
| `quant_log` | `0x72D782` | lossy | numeric -> numeric | `QuantLog(recipe, dtype)` | logarithmic grid, relative bound |
| `quant_sqrt` | `0x72D783` | lossy | numeric -> numeric | `QuantSqrt(recipe, dtype)` | square-root grid, bound follows `sqrt(a + b*x)` |

The README documents seventeen codecs: six predictors, four fused codecs,
`lanes`, `deinterleave`, `nodata`, `pfor` and three quantizers.

## 3. Header layouts

| Codec | Bytes | Layout |
| --- | --- | --- |
| `delta_w` | 4 | `u32 width` |
| `delta_n`, `planar`, `planar_zigzag`, `med`, `med_zigzag`, `average` | 4 or 8 | `u32 width` [`u32 planes`, written only when planes > 1] |
| `wp_static` | 13 or 17 | `u32 width`, `u8 shift` (< 64), `i16 cN, cNW, cNE, cNN` [`u32 planes`] |
| `deinterleave` | 0 | lane counts must be equal or even lane one longer |
| `nodata` | `w` or `4w` | plain: the NoData bit pattern; guarded: the pattern, then its neighbour above, below and the other signed zero |
| `pfor` | 9 | `u64 count`, `u8 element width` |
| `planar_zigzag_pfor` | 17 | `u64 count`, `u8 element width`, `u32 width`, `u32 planes` |
| `planar_zigzag_lanes` | 21 + 7 per lane | `u64 count`, `u8 element width`, `u32 width`, `u32 planes`, `u32 PivCo block size`, then per byte lane `u8 mode` (0 PivCo, 1 PFOR), `u16 weights size`, `u32 payload size` |
| `lanes` | 13 + 7 per lane | `u64 count`, `u8 element width`, `u32 PivCo block size`, then the same per-lane records |
| `quant_linear`, `quant_log` | 10 | `u8 dtype`, `u8 flags`, `f64 step` |
| `quant_sqrt` | 18 | `u8 dtype`, `u8 flags`, `f64 step`, `f64 offset` |
| coefficient blob (not a codec) | 6 + 4 per vector + 4 per value | `"GZC1"`, `u8 version (1)`, `u8 vector count`, then per vector `u32 n` and `n` x `i32`; in the frame header comment |

Quantizer flags: bit 0 `NONNEGATIVE` (clamp at zero); bit 1 `STORE_VALUES` (always set
on integer LOG, set on integer SQRT when the index does not fit, refused everywhere else);
and, for `quant_sqrt`, bit 2 `DECODE_F32` (rebuild f32 indices in binary32).

Width and planes must be nonzero and each plane must hold whole rows, or the frame is
corrupt. PFOR readers bound the declared count by the payload (at most 128 elements per
byte) before allocating.

## 4. Decoding formulas

Arithmetic wraps at the element width unless stated.

| Codec | Reconstruction |
| --- | --- |
| `delta_w` | `out[y, x] = r + out[y, x-1]`, each row starts absolute |
| `delta_n` | `out[y, x] = r + out[y-1, x]`, first row of each plane absolute |
| `planar` | `out = r + W + N - NW` (neighbours outside the plane are 0) |
| `med` | `out = r + med(W, N, NW)` |
| `average` | `out = r + (W >> 1) + (N >> 1) + (W & N & 1)` |
| `wp_static` | `out = r + W + ((cN*N + cNW*NW + cNE*NE + cNN*NN + round) >> shift)`, 32-bit accumulator for 1 and 2 byte samples, 64-bit otherwise |
| `planar_zigzag` | `r = (z >> 1) ^ -(z & 1)`, then planar |
| `med_zigzag` | `r = (z >> 1) ^ -(z & 1)`, then med |
| `planar_zigzag_lanes` | decode each byte lane (PivCo or byte PFOR), `z = sum lane_k << 8k`, then as `planar_zigzag` |
| `lanes` | decode each byte lane (PivCo or byte PFOR), then join the lanes little endian |
| `pfor` | per block: unpack the `b`-bit body, then OR `exception_bits << b` into each listed position |
| `nodata` | plain: `out = mask == 0 ? pattern : values`; guarded (mask 0 hole, 1 above, 2 below, 3 other zero): a hole takes `pattern`, a value equal to `pattern` takes the replacement its code names, anything else is copied; codes above 3 or a replacement equal to `pattern` are corrupt |
| `quant_linear` | integer index: `q * step` in integer arithmetic; float index: `q * step`; clamp to dtype (and to 0 when `NONNEGATIVE`) |
| `quant_log` | float index `q`: `0 -> 0`, else `sign(q) * 2^(A + (abs(q) - 1) * step)` with `A` of -24, -149, -1074 for f16, f32, f64; integer values: copy |
| `quant_sqrt` | index: `(q * step)^2 - offset` (rounded for integers); integer values: copy |

PFOR body layout: 256 values per block interleaved over 16-byte groups
(`L = 16 / W` lanes per group), `b` bits per value LSB first; exceptions listed as a
32-byte bitmap or an ascending position list, whichever the encoder chose. The layout is
fixed across machines. See `core/src/pfor/spec.md` for the full block grammar.

## 5. dtype codes

Part of the wire format (quantizer headers) and of the C API (`geozl_dtype` in `dtype.h`):

| Code | dtype | Code | dtype |
| ---: | --- | ---: | --- |
| 0 | `uint8` | 6 | `int32` |
| 1 | `uint16` | 7 | `int64` |
| 2 | `uint32` | 8 | `float16` |
| 3 | `uint64` | 9 | `float32` |
| 4 | `int8` | 10 | `float64` |
| 5 | `int16` | | |

## 6. Retired CTids

`0x72D708` to `0x72D70B`, `0x72D70E` and `0x72D780` are reserved and have no
decoder. The comments in `ctids.h` keep them from being reused; a frame that names one
fails to find a decoder instead of being misread.
