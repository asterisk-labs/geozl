#ifndef GEOZL_CODECS_QUANT_DECODE_BINDING_H
#define GEOZL_CODECS_QUANT_DECODE_BINDING_H

#include "graph_quant.h"
#include "openzl/zl_dtransform.h" // ZL_Decoder, ZL_TypedDecoderDesc

ZL_Report DI_geozl_quant(ZL_Decoder *dictx, const ZL_Input *ins[]);

#define DI_QUANT(id)                                                           \
  {                                                                            \
    .gd = QUANT_GRAPH, .transform_f = DI_geozl_quant,                          \
    .name = "geozl.lossy.quant",                                               \
  }

#endif // GEOZL_CODECS_QUANT_DECODE_BINDING_H
