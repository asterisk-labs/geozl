#ifndef GEOZL_CODECS_PLANAR_DECODE_BINDING_H
#define GEOZL_CODECS_PLANAR_DECODE_BINDING_H

#include "openzl/zl_dtransform.h" // ZL_Decoder, ZL_TypedDecoderDesc

ZL_Report DI_geozl_planar(ZL_Decoder* dictx, const ZL_Input* ins[]);

extern const ZL_TypedDecoderDesc planar_decoder_desc;

#endif // GEOZL_CODECS_PLANAR_DECODE_BINDING_H
