#ifndef GEOZL_CODECS_DELTA_N_ENCODE_BINDING_H
#define GEOZL_CODECS_DELTA_N_ENCODE_BINDING_H

#include "graph_delta_n.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_delta_n(ZL_Encoder* eictx, const ZL_Input* in);

static const ZL_TypedEncoderDesc delta_n_encoder_desc = {
    .gd          = DELTA_N_GRAPH,
    .transform_f = EI_geozl_delta_n,
    .name        = "geozl.lossless.delta_n",
};

#endif // GEOZL_CODECS_DELTA_N_ENCODE_BINDING_H
