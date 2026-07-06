#ifndef GEOZL_CODECS_DEINTERLEAVE_ENCODE_BINDING_H
#define GEOZL_CODECS_DEINTERLEAVE_ENCODE_BINDING_H

#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_deinterleave(ZL_Encoder* eictx, const ZL_Input* in);

extern const ZL_TypedEncoderDesc deinterleave_encoder_desc;

#endif // GEOZL_CODECS_DEINTERLEAVE_ENCODE_BINDING_H
