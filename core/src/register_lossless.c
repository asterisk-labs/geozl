#include "geozl/geozl.h"

#include "delta_w/decode_delta_w_binding.h"
#include "delta_n/decode_delta_n_binding.h"
#include "planar/decode_planar_binding.h"

#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"

ZL_Report geozl_register_lossless_decoders(ZL_DCtx* dctx)
{
    ZL_Report r = ZL_DCtx_registerTypedDecoder(dctx, &delta_w_decoder_desc);
    if (ZL_isError(r)) return r;
    r = ZL_DCtx_registerTypedDecoder(dctx, &delta_n_decoder_desc);
    if (ZL_isError(r)) return r;
    r = ZL_DCtx_registerTypedDecoder(dctx, &planar_decoder_desc);
    if (ZL_isError(r)) return r;
    return ZL_returnSuccess();
}
