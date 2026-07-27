## Nodata Decoder Specification

### Inputs
Two numeric streams with the same number of elements, `nbElts`.

- `values`, the raster with every missing sample replaced by a fill. Element
  width 1, 2, 4 or 8.
- `mask`, one byte per sample. Following GDAL, `0` marks a sample that was never
  measured and any nonzero byte marks a valid one.

### Codec Header
`1 + eltWidth` bytes, little endian.

- byte 0, the code, `1`. A reader must reject any other value rather than
  interpret the bytes behind it.
- bytes `1..eltWidth`, the bit pattern to restore, at the width of a sample.

### Decoding
The output is `values` with the stored pattern written wherever `mask` is zero.

    out[i] = mask[i] == 0 ? pattern : values[i]

The transform is bit exact. The pattern is stored as bits rather than as a
number, so a NaN comes back with the payload it went in with, and a sentinel
such as -9999 comes back unchanged. One pattern is stored, so a tile carrying
more than one NaN payload comes back with the first of them throughout. Every
NaN is a hole either way, which is what lets a near lossless codec downstream
assume it never sees one. The fill under the mask is discarded, so
what the encoder chose to write there never reaches a reader.
