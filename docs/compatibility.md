# Compatibility

## Frames

**0.14.0 is the baseline.** This release and every later release will read
frames written by 0.14.0. Later readers will also read frames produced by
intervening compatible releases. The 0.13.x line predates this baseline, so its
integer lossy frames are not covered by that promise: 0.14.0 changed what an
integer `quant_linear` frame carries, and a 0.13.x reader refuses such a frame
rather than misreading it, which is the failure this policy exists to guarantee.

Released CTids are never reassigned. An incompatible codec layout receives a
new CTid. Encoders may produce different bytes without breaking compatibility,
and older readers may not understand codecs introduced later.

Released frames in `bindings/python/test/golden` are decoded in CI on x86-64
and arm64.

### Changing the wire format of an existing codec

Adding a codec is covered by [adding a codec](adding-a-codec.md). Changing one
that already ships is a different job, and this is the checklist it earned:

1. Freeze a golden frame in every wire form the change touches, including the
   one being retired, before the change lands.
2. Confirm an older reader **refuses** the new frame rather than misreading it.
   Trace the actual predicate; do not assume.
3. Decide whether the change needs a new CTid or only widens which flag values
   are legal, and record the reasoning here rather than only in the changelog.
4. Say which direction of compatibility survives, in the changelog entry.
5. Count what redundancy the frame loses. A flag that was implied by another
   field used to make a bit flip detectable; once both values are legal it is
   not, and the only thing left is the frame checksum.

## APIs

The C source API is stable from 0.13.0. Breaking changes require a major
release. The C ABI is not stable before 1.0, so C applications should be
recompiled after an update. See [C API stability](c-api.md).

The public Python API follows semantic versioning but may grow before 1.0.

## OpenZL

GeoZL frames are OpenZL frames, so OpenZL container compatibility follows
OpenZL. The Python package supports OpenZL 0.2.x.
