#ifndef GEOZL_CODECS_PLANAR_ENCODE_BINDING_H
#define GEOZL_CODECS_PLANAR_ENCODE_BINDING_H

#include "graph_planar.h"
#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_planar(ZL_Encoder* eictx, const ZL_Input* in);

static const ZL_TypedEncoderDesc planar_encoder_desc = {
    .gd          = PLANAR_GRAPH,
    .transform_f = EI_geozl_planar,
    .name        = "geozl.lossless.planar",
};

#endif // GEOZL_CODECS_PLANAR_ENCODE_BINDING_H
