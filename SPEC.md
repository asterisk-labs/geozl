# geozl

- Specification 0.1.0
- Status Draft
- Date 2026-06-30
- License BSD-3-Clause

geozl is a set of custom codecs for OpenZL that compress 2D raster tiles. A tile is one OpenZL frame, and the codecs run inside it. geozl defines codecs, not a container.

A codec is one of two kinds. A lossless codec reconstructs the tile exactly. A near lossless codec reconstructs it within an error bound it declares in the frame.

The words MUST, MUST NOT, SHOULD, and MAY carry the RFC 2119 meaning.

## Element types

A tile is a typed numeric stream of one component type, at an element width of 1, 2, 4, or 8 bytes.

| code | width | encoding |
|---|---|---|
| 0 | 1 | unsigned 8-bit integer |
| 1 | 2 | unsigned 16-bit integer |
| 2 | 4 | unsigned 32-bit integer |
| 3 | 8 | unsigned 64-bit integer |
| 4 | 1 | signed 8-bit integer |
| 5 | 2 | signed 16-bit integer |
| 6 | 4 | signed 32-bit integer |
| 7 | 8 | signed 64-bit integer |
| 8 | 2 | IEEE 16-bit float |
| 9 | 4 | IEEE 32-bit float |
| 10 | 8 | IEEE 64-bit float |

OpenZL has no complex type, a tile is always one of the real component types above. A complex value is two reals, so geozl carries it as its real component type with twice the element count, the real and imaginary parts either interleaved or split into two streams. The split is done by the `complex_split` codec, which takes the real component stream and emits the two parts as separate streams, so a spatial codec after it never crosses the component boundary.

## Transform ids

A geozl codec is an OpenZL custom transform. The id is the codec's identity on the wire, it travels in the frame and the decoder dispatches on it, so a reader MUST have the codec registered to decode a frame that uses it, and the id MUST be unique among the codecs registered in one decoder.

OpenZL numbers its standard transforms from the low end, near zero. geozl takes its ids from `0x72D700` to `0x72D7FF`, a block high enough that OpenZL's standard ids, which grow upward from zero, will not reach it. The block is split by kind, `0x72D700` to `0x72D77F` lossless and `0x72D780` to `0x72D7FF` near lossless, so a reader tells the kind from the id without decoding the payload.

geozl treats a published id as permanent. Its decode format is frozen and later geozl keeps the decoder for it, so a frame written today stays readable. New behaviour takes a new id, an existing one never changes.

## Lossless codecs

A lossless codec satisfies `decode(encode(x)) = x`. It carries in its codec header whatever the decoder needs to invert it, if anything.

## Near lossless codecs

A near lossless codec quantizes the tile to a signed integer index at the source element width. It does not satisfy `decode(encode(x)) = x`, but bounds its reconstruction error and declares one error mode, the guarantee it makes about that bound.

- **One boundary.** A near lossless frame MUST carry exactly one near lossless codec, as the head transform. The loss happens once, every stage after it is lossless.
- **Header.** It begins with the component type code, so a reader knows the element type even for a codec it does not implement, and the error parameters follow. A reader reports the bound from the head id and this header without decoding the payload.
- **Finite input.** The input MUST NOT contain NaN or Inf, nodata is handled before the codec.

## Changelog

- 0.1.0. Initial draft.