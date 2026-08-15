# Compatibility

## Frames

Starting with 0.13.0, every GeoZL release will read frames written by earlier
releases from 0.13.0 onward.

Released CTids are never reassigned. An incompatible codec layout receives a
new CTid. Encoders may produce different bytes without breaking compatibility,
and older readers may not understand codecs introduced later.

Released frames in `bindings/python/test/golden` are decoded in CI on x86-64
and arm64.

## APIs

The C source API is stable from 0.13.0. Breaking changes require a major
release. The C ABI is not stable before 1.0, so C applications should be
recompiled after an update. See [C API stability](c-api.md).

The public Python API follows semantic versioning but may grow before 1.0.

## OpenZL

GeoZL frames are OpenZL frames, so OpenZL container compatibility follows
OpenZL. The Python package supports OpenZL 0.2.x.
