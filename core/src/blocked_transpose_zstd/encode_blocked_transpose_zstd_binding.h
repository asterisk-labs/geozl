#ifndef GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_ENCODE_BINDING_H
#define GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_ENCODE_BINDING_H

#include "graph_blocked_transpose_zstd.h"
#include "openzl/zl_ctransform.h"

ZL_Report EI_geozl_blocked_transpose_zstd(ZL_Encoder *eictx,
                                           const ZL_Input *in);

#define EI_BLOCKED_TRANSPOSE_ZSTD(id)                                          \
  {                                                                            \
    .gd = BLOCKED_TRANSPOSE_ZSTD_GRAPH(id),                                    \
    .transform_f = EI_geozl_blocked_transpose_zstd,                            \
    .name = "geozl.lossless.blocked_transpose_zstd",                          \
  }

#endif // GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_ENCODE_BINDING_H
