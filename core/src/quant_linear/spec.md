## Quant-Linear Decoder Specification

The 'quant_linear' codec is lossy, in the `geozl.lossy` namespace, CTid `0x72D780`. It
is uniform near lossless quantization. The encoder rounds to a step with
`q = round(x / step)`, which discards the within step remainder, so the decoder does
not reconstruct the original stream exactly. The error is bounded, see Error bound.

The stream carries one of two things, and the sign of `scale` says which. A positive
`scale` means it carries the indices `q` and the decoder multiplies them back. A
negative `scale` means the encoder already multiplied, so the stream carries the
reconstruction and the decoder only copies it. Both give the same output for the same
input. The stored reconstruction skips a pass on decode and costs a little ratio,
because the indices are smaller numbers than the values they stand for.

It works on unsigned integer, signed integer, and IEEE float of 16, 32, or 64 bits.
The original element type is not carried by the numeric stream, which only knows its
width, so it travels in the codec header, the same way OpenZL's float_deconstruct
carries its element type.

### Inputs
The decoder takes a single numeric stream. Its element width is the same as the
original element width, 1, 2, 4, or 8 bytes.

With a positive `scale` the stream holds indices. For integer originals the index is
the same integer type. For float originals the index is a signed integer of the same
byte width, f16 to int16, f32 to int32, f64 to int64.

With a negative `scale` the stream holds the reconstruction, in the original integer
type. Only the integer types can do this, since a float reconstruction is not the
integer stream the codec emits, so a negative `scale` with a float type is corrupt.

### Codec Header
The codec header is 9 bytes, little-endian.

- byte 0, the original element type, an enum.
  - 0 u8, 1 u16, 2 u32, 3 u64
  - 4 i8, 5 i16, 6 i32, 7 i64
  - 8 f16, 9 f32, 10 f64
- bytes 1 to 8, the quantization step `scale`, an IEEE double, where the magnitude is
  the step and the sign picks what the stream holds. `|scale| = 2 * max_error`. For
  integer originals the magnitude is an integer value stored in the double.
  - `scale > 0`, the stream holds indices, the decoder multiplies.
  - `scale < 0`, the stream holds the reconstruction, the decoder copies. Integer
    types only.
  - `scale == 0`, an exact passthrough.

  `inf` and `nan` are corrupt. So is a negative `scale` on a float type.

### Decoding
The decode reads the type and the scale from the header, then reconstructs each value.

- `scale > 0`, integer original, `x = q * scale`, clamped to the type range so the top
  does not wrap.
- `scale > 0`, float original, `x = q * scale` computed in double and cast to the float
  width.
- `scale < 0`, the stream is already the answer, copy it.
- `scale == 0`, copy it.

On integers a `scale` of 1 is also a copy, since the index and the value are the same
number.

Consider an integer stream {0, 1, 3} with type u16 and scale 10. The decoded stream is
{0, 10, 30} as u16. The same tile with scale -10 arrives as {0, 10, 30} already and is
copied through.

### Outputs
A single numeric stream of the original element type and width, with the same number of
elements as the input.

### Error bound
The encoder rounds each value to the nearest multiple of the step, and the step is
`2 * max_error`, so every decoded value is within `max_error` of the original. The
bound does not depend on the sign of `scale`, both forms reconstruct the same values.
For
integer types this is exact. For float types the decoded value is `q * scale` rounded
to the float width, which can add up to half a ULP of that float type on top of
`max_error`. This extra is inherent to the float representation and negligible.

The step applies to the whole tile. There is no per block minimum, because the geozl
spatial predictors remove the local offset, so the per block minimum that LERC needs is
not used here.
