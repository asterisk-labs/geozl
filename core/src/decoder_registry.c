#include "geozl/geozl.h"

#include "average/decode_average_binding.h"
#include "deinterleave/decode_deinterleave_binding.h"
#include "delta_n/decode_delta_n_binding.h"
#include "delta_w/decode_delta_w_binding.h"
#include "med/decode_med_binding.h"
#include "planar/decode_planar_binding.h"
#include "quant_linear/decode_quant_linear_binding.h"
#include "wp_static/decode_wp_static_binding.h"

#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"

#include <stddef.h>

ZL_Report geozl_register_decoders(ZL_DCtx* dctx)
{
    static const ZL_TypedDecoderDesc* const decoders[] = {
        &delta_w_decoder_desc,      &delta_n_decoder_desc,
        &planar_decoder_desc,       &med_decoder_desc,
        &average_decoder_desc,      &wp_static_decoder_desc,
        &deinterleave_decoder_desc, &quant_linear_decoder_desc,
    };
    for (size_t i = 0; i < sizeof(decoders) / sizeof(decoders[0]); ++i) {
        ZL_Report r = ZL_DCtx_registerTypedDecoder(dctx, decoders[i]);
        if (ZL_isError(r))
            return r;
    }
    return ZL_returnSuccess();
}

int geozl_owns_ctid(uint32_t ctid)
{
    return ctid >= GEOZL_CTID_FIRST && ctid <= GEOZL_CTID_LAST;
}

int geozl_ctid_is_lossy(uint32_t ctid)
{
    return ctid > GEOZL_CTID_LOSSLESS_LAST && ctid <= GEOZL_CTID_LAST;
}