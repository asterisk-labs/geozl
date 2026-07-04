#ifndef GEOZL_CODECS_WP_STATIC_ENCODE_BINDING_H
#define GEOZL_CODECS_WP_STATIC_ENCODE_BINDING_H

#include "graph_wp_static.h"
#include "openzl/zl_ctransform.h"

ZL_Report EI_geozl_wp_static(ZL_Encoder* eictx, const ZL_Input* in);

static const ZL_TypedEncoderDesc wp_static_encoder_desc = {
    .gd          = WP_STATIC_GRAPH,
    .transform_f = EI_geozl_wp_static,
    .name        = "geozl.lossless.wp_static",
};

#endif // GEOZL_CODECS_WP_STATIC_ENCODE_BINDING_H
