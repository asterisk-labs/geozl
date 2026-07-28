#ifndef GEOZL_CODECS_QUANT_ENCODE_BINDING_H
#define GEOZL_CODECS_QUANT_ENCODE_BINDING_H

#include "graph_quant.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_quant(ZL_Encoder *eictx, const ZL_Input *in);

#define EI_QUANT(id)                                                           \
  {                                                                            \
    .gd = QUANT_GRAPH, .transform_f = EI_geozl_quant,                          \
    .name = "geozl.lossy.quant",                                               \
  }

#endif // GEOZL_CODECS_QUANT_ENCODE_BINDING_H
