#ifndef GEOZL_CODECS_QUANT_LINEAR_DECODE_BINDING_H
#define GEOZL_CODECS_QUANT_LINEAR_DECODE_BINDING_H

#include "openzl/zl_dtransform.h" // ZL_Decoder, ZL_TypedDecoderDesc

ZL_Report DI_geozl_quant_linear(ZL_Decoder* dictx, const ZL_Input* ins[]);

extern const ZL_TypedDecoderDesc quant_linear_decoder_desc;

#endif // GEOZL_CODECS_QUANT_LINEAR_DECODE_BINDING_H
