## Int-Mult Decoder Specification

### Inputs
The decoder for the 'intmult' codec takes two numeric streams with the same
number of elements, `nbElts`, and the same element width.

- `mults`, `mults[k]` is the quotient of the original value by the base.
- `adjs`, `adjs[k]` is the remainder of the original value by the base.

### Codec Header
Little endian, the `base` as `eltWidth` bytes, where `eltWidth` is the element
width of the streams. `base` must be at least 2. The header size must be
exactly `eltWidth`.

### Decoding
Reading each element as an unsigned integer of the stream's width, the `nth`
output element is

    u = mults[n] * base + adjs[n]      (wrapping, modulo 2^(8*eltWidth))

The pair is valid only when `adjs[n] < base`. The decoder must reject a frame
containing an invalid pair as corrupt. The multiply wraps for any `mults[n]`,
which is deterministic and memory safe, the frame checksum catches a corrupt
quotient.

The encoder guarantees `mults[n] = u / base` and `adjs[n] = u % base` for the
original `u`, so `mults[n] * base` never overflows for genuine data and the
round trip is exact.
