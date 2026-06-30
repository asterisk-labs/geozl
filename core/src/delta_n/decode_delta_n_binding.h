#ifndef GEOZL_CODECS_DELTA_N_DECODE_BINDING_H
#define GEOZL_CODECS_DELTA_N_DECODE_BINDING_H

#include "graph_delta_n.h"
#include "openzl/zl_dtransform.h" // ZL_Decoder, ZL_TypedDecoderDesc

ZL_Report DI_geozl_delta_n(ZL_Decoder* dictx, const ZL_Input* ins[]);

static const ZL_TypedDecoderDesc delta_n_decoder_desc = {
    .gd          = DELTA_N_GRAPH,
    .transform_f = DI_geozl_delta_n,
    .name        = "geozl.lossless.delta_n",
};

#endif // GEOZL_CODECS_DELTA_N_DECODE_BINDING_H
