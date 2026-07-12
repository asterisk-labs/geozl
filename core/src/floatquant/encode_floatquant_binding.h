#ifndef GEOZL_CODECS_FLOATQUANT_ENCODE_BINDING_H
#define GEOZL_CODECS_FLOATQUANT_ENCODE_BINDING_H

#include "common/graph_num1to2.h"
#include "openzl/zl_ctransform.h"

ZL_Report EI_geozl_floatquant(ZL_Encoder *eictx, const ZL_Input *in);

#define EI_FLOATQUANT(id)                                                      \
  {                                                                            \
    .gd = GEOZL_NUM1TO2_GRAPH(id), .transform_f = EI_geozl_floatquant,         \
    .name = "geozl.lossless.pco_floatquant",                                   \
  }

#endif // GEOZL_CODECS_FLOATQUANT_ENCODE_BINDING_H
