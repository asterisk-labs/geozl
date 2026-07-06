#ifndef GEOZL_CODECS_DELTA_W_ENCODE_BINDING_H
#define GEOZL_CODECS_DELTA_W_ENCODE_BINDING_H

#include "openzl/zl_ctransform.h" // ZL_Encoder, ZL_TypedEncoderDesc

ZL_Report EI_geozl_delta_w(ZL_Encoder* eictx, const ZL_Input* in);

// Descriptor for ZL_Compressor_registerTypedEncoder. Defined in the .c so the
// object has a single definition and raises no unused warning in a translation
// unit that only needs the function.
extern const ZL_TypedEncoderDesc delta_w_encoder_desc;

#endif // GEOZL_CODECS_DELTA_W_ENCODE_BINDING_H
