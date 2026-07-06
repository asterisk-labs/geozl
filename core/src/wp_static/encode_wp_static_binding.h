#ifndef GEOZL_CODECS_WP_STATIC_ENCODE_BINDING_H
#define GEOZL_CODECS_WP_STATIC_ENCODE_BINDING_H

#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_wp_static(ZL_Encoder* eictx, const ZL_Input* in);

extern const ZL_TypedEncoderDesc wp_static_encoder_desc;

#endif // GEOZL_CODECS_WP_STATIC_ENCODE_BINDING_H
