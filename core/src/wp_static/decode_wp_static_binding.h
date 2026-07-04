#ifndef GEOZL_CODECS_WP_STATIC_DECODE_BINDING_H
#define GEOZL_CODECS_WP_STATIC_DECODE_BINDING_H

#include "graph_wp_static.h"
#include "openzl/zl_dtransform.h"

ZL_Report DI_geozl_wp_static(ZL_Decoder* dictx, const ZL_Input* ins[]);

static const ZL_TypedDecoderDesc wp_static_decoder_desc = {
    .gd          = WP_STATIC_GRAPH,
    .transform_f = DI_geozl_wp_static,
    .name        = "geozl.lossless.wp_static",
};

#endif // GEOZL_CODECS_WP_STATIC_DECODE_BINDING_H
