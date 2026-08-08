## Nodata Decoder Specification

### Inputs
Two numeric streams, both `nbElts` elements long.

- `values`, the raster with every missing sample replaced by a fill. Element
  width 1, 2, 4 or 8.
- `mask`, one byte per sample. Following GDAL, `0` marks a sample that was never
  measured and any nonzero byte marks a valid one.

`nbElts` is the length of `values`, and a reader must reject a frame where
`mask` is a different length or where either is empty.

### Codec Header
`eltWidth` bytes, little endian, the bit pattern to restore at the width of a
sample. There is nothing else in it.

No code, since there is one shape, and no count, since `values` carries it.
Nothing the frame declares reaches an allocation.

An encoder must refuse an empty tile, which has no mask to carry.

### Decoding
The output is `values` with the stored pattern written wherever `mask` is zero.

    out[i] = mask[i] == 0 ? pattern : values[i]

The pattern is stored as bits rather than as a number, so a NaN comes back with
the payload it went in with, and a sentinel such as -9999 comes back unchanged.
The fill under the mask is discarded, so what the encoder chose to write there
never reaches a reader.

The transform is bit exact on every tile that carries at most one NaN payload,
which is the only case the encoder can restore. Only one pattern is stored, so a
tile carrying a second payload comes back with the first one in its place. Every
NaN is marked either way, which is what lets a near lossless codec downstream
assume it never sees one.

The encoder marks a sentinel by comparing bit patterns, while GDAL compares
numerically. The two agree everywhere except on `-0.0`, which GDAL counts as
equal to `0.0` and this codec does not.
