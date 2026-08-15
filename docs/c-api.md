# C API stability

Beginning with 0.13.0, the declarations in `geozl/geozl.h` are a stable source
API. Existing functions and parameter lists will remain available; new entry
points may be added. A breaking change requires a major release.

Binary compatibility is not promised before 1.0. Recompile C applications when
updating GeoZL or OpenZL.
