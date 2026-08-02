## Nodata Decoder Specification

### Inputs
Two numeric streams.

- `values`, the raster with every missing sample replaced by a fill. Element
  width 1, 2, 4 or 8.
- `mask`, one byte per sample. Following GDAL, `0` marks a sample that was never
  measured and any nonzero byte marks a valid one.

How many elements each stream holds depends on the header code below, and a
reader must check the sizes against the code rather than trust it alone.

### Codec Header
Little endian. Byte 0 is the code, and a reader must reject a code it does not
know rather than interpret the bytes behind it.

| code | name | header size | `values` | `mask` |
| --- | --- | --- | --- | --- |
| 1 | restore | `1 + eltWidth` | `nbElts` | `nbElts` |
| 2 | all valid | `1` | `nbElts` | empty |
| 3 | all hole | `1 + eltWidth + 8` | empty | empty |

For codes 1 and 3, bytes `1..eltWidth` hold the bit pattern to restore, at the
width of a sample. Code 3 drops both streams, so the sample count has nowhere
else to live and follows the pattern as a uint64. That count sizes an
allocation, so a reader must reject a value that overflows when multiplied by
the element width.

### Decoding

Code 1, the output is `values` with the stored pattern written wherever `mask`
is zero.

    out[i] = mask[i] == 0 ? pattern : values[i]

Code 2, nothing was missing, so the output is `values` unchanged. Code 3,
nothing was measured, so the output is the stored pattern repeated `nbElts`
times.

The transform is bit exact. The pattern is stored as bits rather than as a
number, so a NaN comes back with the payload it went in with, and a sentinel
such as -9999 comes back unchanged. The fill under the mask is discarded, so
what the encoder chose to write there never reaches a reader.

One pattern is stored, so a tile carrying more than one NaN payload comes back
with the first of them throughout. Every NaN is a hole either way, which is what
lets a near lossless codec downstream assume it never sees one.

The encoder marks a sentinel by comparing bit patterns, while GDAL compares
numerically. The two agree everywhere except on `-0.0`, which GDAL counts as
equal to `0.0` and this codec does not.
