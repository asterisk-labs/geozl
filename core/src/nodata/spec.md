# nodata Decoder Specification

Lossless numeric codec, CTID `0x72D70C`.

## Inputs

Two nonempty streams with the same element count:

- `values`, with elements 1, 2, 4, or 8 bytes wide.
- `mask`, with one-byte elements. Zero marks nodata; nonzero marks valid data.

## Codec header

Exactly `eltWidth` little-endian bytes containing the nodata bit pattern.

## Decoding

    out[i] = mask[i] == 0 ? pattern : values[i]

The comparison and restoration use bit patterns. This distinguishes `-0.0`
from `0.0` and preserves a NaN payload. Because the header stores one pattern,
all masked NaNs are restored with the first stored payload.

## Output

One numeric stream with the width and element count of `values`.
