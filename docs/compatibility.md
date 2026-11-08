# Compatibility

## Frames

GeoZL makes no compatibility promise before 1.0. A release may change a codec
layout, and its readers may refuse frames that earlier releases wrote.

Released CTids are never reassigned. An incompatible codec layout receives a
new CTid. Encoders may produce different bytes without breaking compatibility,
and older readers may not understand codecs introduced later.

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
