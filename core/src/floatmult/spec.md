## Float-Mult Decoder Specification

### Inputs
Two numeric streams with the same element count `nbElts` and element width, 4
(f32) or 8 (f64).

- `primary`, `mult` stored via the integer-float latent, the nearest multiple
  count of the base.
- `secondary`, the centered ULP adjustment between the value and `mult * base`.

### Codec Header
The `base` as an IEEE f64, 8 bytes little endian, used directly for f64 and
narrowed to f32 for the width-4 path. `base` must be nonzero and finite.

### Decoding
Let `ord` be the total order key of the width and `MID` its sign bit. For each
element, decoding the integer-float latent `primary` to the float `mult`,

    prod = mult * base
    key  = ord(prod) + (secondary - MID)     (wrapping)
    x    = ord_inverse(key)

`mult * base` must be computed in the stream's own precision (f32 for width 4)
so it matches the encoder bit for bit. The transform is bit exact for every
float including NaN and infinities, which the encoder maps through a `mult` of
zero. There is no per element validity check; corruption is caught by the frame
checksum.
