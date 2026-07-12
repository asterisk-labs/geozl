## Float-Quant Decoder Specification

### Inputs
Two numeric streams with the same number of elements, `nbElts`, and the same
element width, which is 4 (f32) or 8 (f64).

- `primary`, the order preserving key of the float with its low `k` bits removed.
- `secondary`, the low `k` mantissa bits, flipped for negative floats.

### Codec Header
A single byte `k`, the number of quantized mantissa bits, in
`1..=PRECISION_BITS` (23 for f32, 52 for f64).

### Decoding
Let `SIGN` be the sign bit of the width (`0x80000000` for f32,
`0x8000000000000000` for f64) and `maxlow = (1 << k) - 1`. For each element,
with `is_pos = primary >= (SIGN >> k)`,

    low = is_pos ? secondary : maxlow - secondary
    key = (primary << k) + low
    bits = (key & SIGN) ? key ^ SIGN : ~key

and the output float is the reinterpretation of `bits`. The pair is valid only
when `secondary <= maxlow`. The decoder must reject a frame with an invalid
pair as corrupt. The transform is bit exact for every float including NaN and
infinities.
