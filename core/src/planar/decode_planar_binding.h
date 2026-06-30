#ifndef GEOZL_CODECS_PLANAR_DECODE_BINDING_H
#define GEOZL_CODECS_PLANAR_DECODE_BINDING_H

#include "graph_planar.h"
#include "openzl/zl_dtransform.h" // ZL_Decoder, ZL_TypedDecoderDesc

ZL_Report DI_geozl_planar(ZL_Decoder* dictx, const ZL_Input* ins[]);

static const ZL_TypedDecoderDesc planar_decoder_desc = {
    .gd          = PLANAR_GRAPH,
    .transform_f = DI_geozl_planar,
    .name        = "geozl.lossless.planar",
};

#endif // GEOZL_CODECS_PLANAR_DECODE_BINDING_H
