#ifndef GEOZL_CODECS_AVERAGE_ENCODE_BINDING_H
#define GEOZL_CODECS_AVERAGE_ENCODE_BINDING_H

#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_average(ZL_Encoder* eictx, const ZL_Input* in);

// Descriptor for ZL_Compressor_registerTypedEncoder.
extern const ZL_TypedEncoderDesc average_encoder_desc;

#endif // GEOZL_CODECS_AVERAGE_ENCODE_BINDING_H
