## Quant Decoder Specification

The 'quant' codec is lossy, in the `geozl.lossy` namespace, CTid `0x72D780`. It
quantizes uniformly in a warped domain, so the error it holds follows a curve
rather than being flat.

A quantizer that respects a pointwise bound `b(x)` is a uniform quantizer in a
domain `w` with `w' = 1/b`. Each curve is that integral for one way the
measurement error of the data grows with the value, so the curve is not a free
choice, it follows from the sensor.

| curve | `b(x)` | `w(x)` | fits |
|---|---|---|---|
| 0 linear | constant | `x` | instrument error, elevation, temperature |
| 1 sqrt | `~ sqrt(x)` | `2*sqrt(x + offset)` | photon counting, optical radiance |
| 2 log | `~ x` | `ln(x)` | multiplicative error, SAR backscatter, concentrations |

### Inputs
A single numeric stream of indices. Its element width is the same as the
original element width, 1, 2, 4 or 8 bytes. For integer originals the index is
the same integer type. For float originals it is a signed integer of the same
byte width, f16 to int16, f32 to int32, f64 to int64.

### Codec Header
The codec header is 27 bytes, little-endian.

- byte 0, the original element type, an enum.
  - 0 u8, 1 u16, 2 u32, 3 u64
  - 4 i8, 5 i16, 6 i32, 7 i64
  - 8 f16, 9 f32, 10 f64
- byte 1, the curve, 0 linear, 1 sqrt, 2 log.
- byte 2, flags. Bit 0 means the stream holds the reconstruction rather than
  the index, so the decoder only copies. The linear curve on an integer type
  only, elsewhere the reconstruction is not the integer stream the codec emits.
- bytes 3 to 10, `step`, an IEEE double. Zero is an exact passthrough.
- bytes 11 to 18, `offset`, an IEEE double.
- bytes 19 to 26, `nsub`, a uint64, the log curve only.

`inf` and `nan` in either double are corrupt, so is a negative `step`, a curve
above 2, a flag bit that is not bit 0, a stored reconstruction outside the
linear curve on an integer type, and a non-zero `nsub` outside the log curve.

### Decoding
- linear, `x = q * step`, clamped to the type range so the top does not wrap.
- sqrt, `x = (q * step)^2 - offset`.
- log, zero decodes to zero. Otherwise with `m = |q|` and `sub = offset/(nsub+1)`,
  `x = sign(q) * m * sub` when `m <= nsub`, and
  `x = sign(q) * offset * exp((m - nsub - 1) * step)` above.

On integers a `step` of 1 under the linear curve is a copy, since the index and
the value are the same number.

The log curve reconstructs through a table of one value per index when the index
range fits in 8192 entries, which on the data this curve is meant for it always
does, so no transcendental is evaluated per sample.

### Outputs
A single numeric stream of the original element type and width, with the same
number of elements as the input.

### Error bound
The bound is a property of the curve, and the encoder holds to it by picking the
index whose reconstruction, taken back at the output width, is nearest the
sample, rather than by trusting the algebra. That is what keeps a declared bound
from depending on how accurate `log` and `exp` happen to be.

- linear, `|x - x^| <= step/2`.
- sqrt, `|x - x^| <= step * sqrt(x + offset)`.
- log, `|x - x^| <= (exp(step/2) - 1) * |x|`.

Two cases sit outside the grid and are carried exactly instead, which a relative
bound allows because it is also satisfied when the two values are identical.
Zero is one, and it decodes to zero for any parameters. The other is the range
where the representable values sit too far apart for any grid to meet the bound,
which is below the smallest normal on a float type and below roughly `8/b` on an
integer one. That range is the leading `nsub` indices, spaced by `offset/(nsub+1)`,
and it is the case that defeats rounding the mantissa to a fixed number of bits.

The reconstruction is rounded once more when it is stored at the output width,
so the grid is cut by that much when the parameters are resolved and the bound
above still holds after the cast.

### Parameters
The parameters are resolved before the graph is built, from the error recipe and
a scan of the tile, because the log curve anchors its grid on the smallest
magnitude present. A tile whose bound needs more levels than the index width
holds is refused rather than saturated.
