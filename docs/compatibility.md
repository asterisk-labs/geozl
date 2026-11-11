# Compatibility

## Frames

GeoZL makes no compatibility promise before 1.0. A release may change a codec
layout, and its readers may refuse frames that earlier releases wrote.

Released CTids are never reassigned to another codec. An incompatible codec
layout receives a new CTid, unless it replaces the old layout in place before
1.0, as recorded below. Encoders may produce different bytes without breaking
compatibility, and older readers may not understand codecs introduced later.

CTids `0x72D708` through `0x72D70B`, `0x72D70E` and `0x72D780` are retired and
reserved. Current readers reject frames using them.

The golden corpus in `bindings/python/test/golden` holds one frame per wire
layout the current readers decode, written by the current code with
`tools/write_golden.py`. CI decodes it on x86-64 and arm64, and the R, Julia
and WASM tests decode the same frames.

Every codec with an encoder is covered in each header form and storage mode it
writes. A layout that is removed loses its frame with it.

### Changing the wire format of an existing codec

Adding a codec is covered by [adding a codec](adding-a-codec.md). Changing one
that already ships needs these steps:

1. Regenerate the golden corpus with `tools/write_golden.py`, so it holds one
   frame per wire layout the changed readers decode.
2. Confirm an older reader **refuses** the new frame rather than misreading it.
3. Decide whether the change needs a new CTid or only widens which flag values
   are legal, and record the reasoning here rather than only in the changelog.
4. Say which direction of compatibility survives, in the changelog entry.
5. Count what redundancy the frame loses. A flag that was implied by another
   field used to make a bit flip detectable; once both values are legal it is
   not, and the only thing left is the frame checksum.

### Layouts changed in place

`planar_zigzag_pfor`, CTid `0x72D710`, changed its payload layout in place. It
keeps its CTid, its name, the `planar>zigzag>pfor` recipe and its 17-byte codec
header. Each plane is now cut into bands of 16 rows and each band into strips
of 1024 columns, the prediction restarts at every band and strip, each
PFOR block is a two-dimensional tile of a band, and the three header bytes of
every block come first, as `core/src/planar_zigzag_pfor/spec.md` describes.
The earlier layout was `planar_zigzag` followed by `pfor`, Zigzag planar
residuals in raster order cut into 256-sample blocks. Frames written with it no
longer decode, apart from the small case below, and no reader keeps a decoder
for them.

It kept its CTid because GeoZL is still in development and no frames of the
old layout need to survive, so a second CTid, a legacy decoder and a second
codec name would cost more than they protect. The golden corpus holds only the
new layout.

Neither direction of compatibility survives, and both readers find a decoder
under the same CTid, so only their payload checks and the frame checksums tell
the layouts apart. Readers on either side refuse nearly every frame of the
other layout, because the block headers at the front throw off the walk
through the payload. A frame that fits in one block and has exceptions parses
as valid in both layouts. It still decodes correctly when it is a single row,
but a frame of several rows decodes to wrong samples. The content checksum
catches that on a CPU, and a GPU, which never checks it, does not.

## APIs

The C source API in `geozl/geozl.h` and `geozl/gpu.h`, the C ABI and the Python
API may change in any release before 1.0. Recompile C applications after an
update. See [C API stability](c-api.md).

## OpenZL

GeoZL frames are OpenZL frames, so OpenZL container compatibility follows
OpenZL. The Python package supports OpenZL 0.3.x.

GeoZL writes OpenZL frame format 27 for every recipe, so its frames need a
reader on OpenZL 0.3. `lanes` and `planar_zigzag_lanes` call the kernels of the pinned OpenZL,
so a change to that bitstream would need another CTid here.
