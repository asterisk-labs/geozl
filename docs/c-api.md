# C API stability

Beginning with 0.13.0, the declarations in `geozl/geozl.h` are a stable source
API, apart from these changes:

- 0.17.0: `geozl_node_nodata` gained the `dtype` and `radius` parameters.
- 0.19.0: `geozl_node_blocked_transpose_zstd`, added in 0.16.0, was removed.
  `geozl_node_binoffset`, `geozl_node_intmult`, `geozl_node_floatquant` and
  `geozl_node_floatmult` began returning `ZL_NODE_ILLEGAL`.
- 0.20.0: the enumerator `GEOZL_CTID_BLOCKED_TRANSPOSE_ZSTD` was removed.
- After 0.20.0: those four functions were removed, with the enumerators
  `GEOZL_CTID_BINOFFSET`, `GEOZL_CTID_INTMULT`, `GEOZL_CTID_FLOATQUANT` and
  `GEOZL_CTID_FLOATMULT`.

A further breaking change requires a major release.

Binary compatibility is not promised before 1.0. Recompile C applications when
updating GeoZL or OpenZL.
