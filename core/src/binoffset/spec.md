## Bin-Offset Decoder Specification

### Inputs
The decoder for the 'binoffset' codec takes two numeric streams with the same
number of elements, `nbElts`.

- `bins`, element width 1. `bins[k]` selects an entry of the bin table.
- `offsets`, element width 1, 2, 4 or 8, the width of the original stream.

### Codec Header
Little endian. One `uint8` holding `nbBins` in 1..=255, then `nbBins` entries
of `{ lower : eltWidth bytes, offset_bits : uint8 }`, where `eltWidth` is the
element width of the offsets stream. `offset_bits` must not exceed
`8 * eltWidth`. The header size must be exactly `1 + nbBins * (eltWidth + 1)`.

### Decoding
Reading each offset as an unsigned integer of the stream's width, the `nth`
output element is

    v = lower[bins[n]] + offsets[n]      (wrapping, modulo 2^(8*eltWidth))

The pair is valid only when `bins[n] < nbBins` and `offsets[n] <
1 << offset_bits[bins[n]]` (any `offsets[n]` when `offset_bits` is the full
width). The decoder must reject a frame containing an invalid pair as corrupt.

The encoder guarantees the table covers every encoded value, sorted ascending
lower bounds with no gaps over the data, so the round trip is exact. The
decoder does not verify table shape beyond the bounds above, overlapping or
gapped tables decode deterministically all the same.
