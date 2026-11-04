# Compatibility

## Frames

**0.20.0 is the first compatibility baseline.** Releases before it were
development releases and their frames are not part of the compatibility
contract. Starting with 0.20.0, later readers will continue to read
frames written by that release and by compatible releases after it.

Released CTids are never reassigned. An incompatible codec layout receives a
new CTid. Encoders may produce different bytes without breaking compatibility,
and older readers may not understand codecs introduced later.

CTids `0x72D708` through `0x72D70B` belonged to pre-baseline, decode-only codecs
ported from pcodec that were removed in 0.21.0. The values remain reserved. Current readers
reject frames using them; no 0.20.0 writer emits them.

Frozen frames in `bindings/python/test/golden` are decoded in CI on x86-64 and
arm64. Frames from before 0.20 are regression fixtures, not a compatibility
promise.

From 0.19, lossy codecs choose the stream layout and recipes no longer accept
`STORE`. The wire format is unchanged, and readers still accept both layouts.
Golden frames preserve the retired 0.14–0.18 value layouts.

### Changing the wire format of an existing codec

Adding a codec is covered by [adding a codec](adding-a-codec.md). Changing one
that already ships needs these steps:

1. Freeze a golden frame in every wire form the change touches, including the
   one being retired, before the change lands.
2. Confirm an older reader **refuses** the new frame rather than misreading it.
3. Decide whether the change needs a new CTid or only widens which flag values
   are legal, and record the reasoning here rather than only in the changelog.
4. Say which direction of compatibility survives, in the changelog entry.
5. Count what redundancy the frame loses. A flag that was implied by another
   field used to make a bit flip detectable; once both values are legal it is
   not, and the only thing left is the frame checksum.

## APIs

The C source API is stable from 0.13.0 apart from `geozl_node_nodata`, which
gained parameters in 0.17.0, `geozl_node_blocked_transpose_zstd`, removed in
0.19.0, and four retired writer stubs and their `GEOZL_CTID_*` enumerators for
CTids `0x72D708` through `0x72D70B`, removed in 0.21.0. The C ABI is not
stable before 1.0, so C applications should be recompiled after an update. See
[C API stability](c-api.md) for the full list.

The public Python API follows semantic versioning but may grow before 1.0.

## OpenZL

GeoZL frames are OpenZL frames, so OpenZL container compatibility follows
OpenZL. The Python package supports OpenZL 0.3.x.

GeoZL writes OpenZL frame format 24, the newest one OpenZL 0.2 reads, so
GeoZL 0.18 readers decode what later releases write. The exceptions are the
`pivco`, `transpose>pivco` and `sparse` terminals. PivCo Huffman exists from
format 27 and `sparse_num` from 26, so their frames need a reader on OpenZL
0.3, and a 0.18 reader refuses them. `planar>zigzag>pivco` writes CTid
`0x72D712`, introduced as `planar_zigzag_pivco` in 0.19 and now named
`planar_zigzag_transpose_pivco`. Its frames need a 0.19 reader.

GeoZL 0.20 writers use CTid `0x72D713` for every non-planar recipe ending in
`pivco` or `transpose>pivco`. The affected predictors are `id`, `delta_1d`,
`delta_w`, `delta_n`, `med`, `average` and `wp_static`. These recipe names
already existed in 0.19, but that release wrote them as standard OpenZL nodes.
Current readers still decode those frames. A 0.19 reader rejects the new CTid,
so upgrade readers before enabling a new writer. Both PivCo codecs call the
kernels of the pinned OpenZL, so a change to that bitstream would need another
CTid here.

Readers accept every format OpenZL 0.3 does, up to 27.
