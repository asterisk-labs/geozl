# C API stability

Beginning with 0.13.0, the declarations in `geozl/geozl.h` are a stable source
API. Four retired writer stubs for CTids `0x72D708` through `0x72D70B` were
removed after 0.20.0; they returned `ZL_NODE_ILLEGAL` and had no writer path.
Other existing functions and parameter lists remain available. A further
breaking change requires a major release.

Binary compatibility is not promised before 1.0. Recompile C applications when
updating GeoZL or OpenZL.
