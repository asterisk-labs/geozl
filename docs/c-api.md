# C API stability

The declarations in `geozl/geozl.h` and `geozl/gpu.h` are the supported C
source API. Before 1.0 they may change in any release, and binary
compatibility is not promised. Recompile C applications when updating GeoZL or
OpenZL.

From 1.0, a breaking change to the source API requires a major release.
