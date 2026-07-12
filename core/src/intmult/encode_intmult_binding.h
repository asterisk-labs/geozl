#ifndef GEOZL_CODECS_INTMULT_ENCODE_BINDING_H
#define GEOZL_CODECS_INTMULT_ENCODE_BINDING_H

#include "common/graph_num1to2.h"
#include "openzl/zl_ctransform.h"

ZL_Report EI_geozl_intmult(ZL_Encoder *eictx, const ZL_Input *in);

#define EI_INTMULT(id)                                                         \
  {                                                                            \
    .gd = GEOZL_NUM1TO2_GRAPH(id), .transform_f = EI_geozl_intmult,            \
    .name = "geozl.lossless.pco_intmult",                                      \
  }

#endif // GEOZL_CODECS_INTMULT_ENCODE_BINDING_H
