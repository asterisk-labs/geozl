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
  Bit 1 means the encoder scanned the tile and found no negative sample, so the
  decoder floors the reconstruction at zero instead of at the type minimum. It
  is measured rather than assumed, and a tile that does hold negative samples
  never carries it.
- bytes 3 to 10, `step`, an IEEE double. Zero is an exact passthrough.
- bytes 11 to 18, `offset`, an IEEE double.
- bytes 19 to 26, `nsub`, a uint64, the log curve only.

`inf` and `nan` in either double are corrupt, so is a negative `step`, a curve
above 2, a flag bit above bit 1, a stored reconstruction outside the linear
curve on an integer type, and a non-zero `nsub` outside the log curve.

### The grid and how the raster is cut
The step comes from the bound the recipe asked for and from the largest magnitude
the tile holds, because storing the reconstruction at the output width rounds it
by a relative eps and the worst case of that is at the largest magnitude. So on a
float type the grid depends on the tile, and two tiles of the same field with
different maxima do not share one. The same physical value can then reconstruct
two ways, up to twice the declared bound apart, which shows up when a raster is
compressed at one chunk size and again at another, and when a single outlying
sample lifts the maximum of a tile and moves the reconstruction of every other
sample in it.

Declaring the range in the recipe removes that. `abs:0.05,max=400` cuts the grid
against 400 rather than against whatever this tile happens to reach, so every
tile of the product resolves to the same grid and the same value always
reconstructs the same way. A pipeline usually has that number already, from the
statistics of the product or from its metadata. `min=` does the same for the
anchor of the log curve and buys back the shorter index that anchoring on the
tile would have given.

Nothing anchors on the tile in the other direction. The linear and log grids both
put zero at a fixed place, and the log anchor comes from the type and the bound
rather than from the smallest magnitude present, so no sample can move it.

### Decoding
- linear, `x = q * step`.
- sqrt, `x = (q * step)^2 - offset`.
- log, zero decodes to zero. Otherwise with `m = |q|` and `sub = offset/(nsub+1)`,
  `x = sign(q) * m * sub` when `m <= nsub`, and
  `x = sign(q) * offset * exp((m - nsub - 1) * step)` above.

Every reconstruction is clamped into the range of the output type, floored at
zero instead when bit 1 of flags is set, and then
rounded to that width before it is stored. It is not only the top that needs
clamping, the sqrt curve lands below zero near the bottom of its range, which on
an unsigned type would wrap. Clamping only ever moves a reconstruction towards
the data, so the bound survives it. The rounding is to nearest, not the
truncation a cast would give, because the encoder picks its index by judging
candidates through the same step.

On integers a `step` of 1 under the linear curve is a copy, since the index and
the value are the same number.

The log curve reconstructs through a table of one value per index when the index
range fits in 8192 entries, so no transcendental is evaluated per sample. The
encoder does the same in reverse for an 8 or 16 bit input, where the whole
forward map fits in a table, and both tables give what the direct arithmetic
would have given.

### Outputs
A single numeric stream of the original element type and width, with the same
number of elements as the input.

### Error bound
The bound is a property of the curve, and the encoder holds to it by picking the
index whose reconstruction, taken back at the output width, is nearest the
sample, rather than by trusting the algebra. That keeps a declared bound off the
accuracy of `log` and `exp`.

It then measures. The encoder decodes what it just wrote, compares every sample
against the bound the recipe declared, and where the frame misses it tightens
the step by the amount it missed by and encodes again, up to twice. A frame that
still misses is refused rather than written. That is what makes the declared
error a measurement rather than an argument, and it holds against causes nobody
predicted. Where the floor is the representation rather than the grid,
tightening does not converge and refusing is the answer.

- linear, `|x - x^| <= step/2`.
- sqrt, `|x - x^| <= step * sqrt(x + offset)`.
- log, `|x - x^| <= (exp(step/2) - 1) * |x|`.

Clamping can only shorten `|x - x^|`, since the interval it clamps into holds
`x`, so the bound above survives it and the floor at zero is free. Without that
floor the sqrt grid is the one curve that can cross zero, because it is anchored
at `-offset` rather than at zero, and a tile of non-negative data comes back
holding small negative values.

Two cases sit outside the grid and are carried exactly instead, which a relative
bound allows because it is also satisfied when the two values are identical.
Zero is one under the linear and log curves, where it is a grid level. The other
is the range
where the representable values sit too far apart for any grid to meet the bound,
which is below the smallest normal on a float type and below roughly `8/b` on an
integer one. That range is the leading `nsub` indices, spaced by
`offset/(nsub+1)`, and it is the case that defeats rounding the mantissa to a
fixed number of bits.

The reconstruction is rounded once more when it is stored at the output width,
half a unit on an integer and a relative eps on a float, so the grid is cut by
that much when the parameters are resolved and the bound above still holds after
the cast. Where the cut is taken depends on the curve. The log curve takes it at
the smallest magnitude, since the tolerance shrinks with the value. The sqrt
curve takes whichever end needs more, since an integer rounding bites hardest at
the bottom and a float rounding at the top.

### Parameters
The parameters are resolved before the graph is built, from the error recipe and
a scan of the tile, because the log curve anchors its grid on the smallest
magnitude present.

Three things are refused rather than encoded into a frame that misses its own
bound. A bound needing more levels than the index width holds, which would
otherwise saturate. A bound smaller than the rounding of a float output type at
the top of the tile, which f16 reaches at ordinary tolerances. And a shot bound
that is under half a unit at the noise floor of an integer type, where no grid
exists and the request is really for lossless.

A sample that is not finite has no index. It encodes as zero, and the nodata
codec in front is what puts it back on the way out.

The index is computed in a double on both sides, so a bound needing more levels
than a double counts exactly, 2^53, is refused however wide the index type is.
Past that point adding one to an index does nothing and the nearest-index search
the bound rests on stops working.