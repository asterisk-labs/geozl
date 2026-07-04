#ifndef GEOZL_CODECS_MED_DECODE_BINDING_H
#define GEOZL_CODECS_MED_DECODE_BINDING_H

#include "graph_med.h"
#include "openzl/zl_dtransform.h"

ZL_Report DI_geozl_med(ZL_Decoder* dictx, const ZL_Input* ins[]);

static const ZL_TypedDecoderDesc med_decoder_desc = {
    .gd          = MED_GRAPH,
    .transform_f = DI_geozl_med,
    .name        = "geozl.lossless.med",
};

#endif // GEOZL_CODECS_MED_DECODE_BINDING_H
