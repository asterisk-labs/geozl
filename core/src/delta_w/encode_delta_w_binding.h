#ifndef GEOZL_CODECS_DELTA_W_ENCODE_BINDING_H
#define GEOZL_CODECS_DELTA_W_ENCODE_BINDING_H

#include "common/graph_num1to1.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_delta_w(ZL_Encoder *eictx, const ZL_Input *in);

// Descriptor for ZL_Compressor_registerTypedEncoder. Defined in the .c so the
// object has a single definition and raises no unused warning in a translation
// unit that only needs the function.

#define EI_DELTA_W(id)                                                         \
  {                                                                            \
    .gd = GEOZL_NUM1TO1_GRAPH(id), .transform_f = EI_geozl_delta_w,            \
    .name = "geozl.lossless.delta_w",                                          \
  }

#endif // GEOZL_CODECS_DELTA_W_ENCODE_BINDING_H
