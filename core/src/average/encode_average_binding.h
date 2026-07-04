#ifndef GEOZL_CODECS_AVERAGE_ENCODE_BINDING_H
#define GEOZL_CODECS_AVERAGE_ENCODE_BINDING_H

#include "graph_average.h"
#include "openzl/zl_ctransform.h"

ZL_Report EI_geozl_average(ZL_Encoder* eictx, const ZL_Input* in);

static const ZL_TypedEncoderDesc average_encoder_desc = {
    .gd          = AVERAGE_GRAPH,
    .transform_f = EI_geozl_average,
    .name        = "geozl.lossless.average",
};

#endif // GEOZL_CODECS_AVERAGE_ENCODE_BINDING_H
