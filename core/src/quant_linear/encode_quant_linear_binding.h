#ifndef GEOZL_CODECS_QUANT_LINEAR_ENCODE_BINDING_H
#define GEOZL_CODECS_QUANT_LINEAR_ENCODE_BINDING_H

#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_quant_linear(ZL_Encoder* eictx, const ZL_Input* in);

extern const ZL_TypedEncoderDesc quant_linear_encoder_desc;

#endif // GEOZL_CODECS_QUANT_LINEAR_ENCODE_BINDING_H
