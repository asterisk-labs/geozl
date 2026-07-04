#ifndef GEOZL_CODECS_AVERAGE_DECODE_BINDING_H
#define GEOZL_CODECS_AVERAGE_DECODE_BINDING_H

#include "graph_average.h"
#include "openzl/zl_dtransform.h"

ZL_Report DI_geozl_average(ZL_Decoder* dictx, const ZL_Input* ins[]);

static const ZL_TypedDecoderDesc average_decoder_desc = {
    .gd          = AVERAGE_GRAPH,
    .transform_f = DI_geozl_average,
    .name        = "geozl.lossless.average",
};

#endif // GEOZL_CODECS_AVERAGE_DECODE_BINDING_H
