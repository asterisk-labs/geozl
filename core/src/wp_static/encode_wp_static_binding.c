#include "encode_wp_static_binding.h"
#include "encode_wp_static_kernel.h"
#include "train_wp_static.h"
#include "header_wp_static.h"

#include "openzl/zl_ctransform.h"
#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>

ZL_Report EI_geozl_wp_static(ZL_Encoder* eictx, const ZL_Input* in)
{
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_numeric);

    const size_t eltWidth = ZL_Input_eltWidth(in);
    const size_t nbElts   = ZL_Input_numElts(in);

    ZL_IntParam wp = ZL_Encoder_getLocalIntParam(eictx, WP_STATIC_PARAM_WIDTH);
    const uint32_t width = (wp.paramId == WP_STATIC_PARAM_WIDTH)
            ? (uint32_t)wp.paramValue
            : (uint32_t)nbElts;

    // The kernel is estimated here, not signaled from the graph builder.
    int16_t coeffs[4];
    uint8_t shift;
    wp_static_train(coeffs, &shift, ZL_Input_ptr(in), width, nbElts, eltWidth);

    uint8_t hdr[WP_STATIC_HEADER_SIZE];
    wp_static_write_header(hdr, width, coeffs, shift);
    ZL_Encoder_sendCodecHeader(eictx, hdr, sizeof(hdr));

    ZL_Output* out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
    if (out == NULL)
        return ZL_returnError(ZL_ErrorCode_allocation);

    wp_static_encode(ZL_Output_ptr(out), ZL_Input_ptr(in), width, nbElts,
                     eltWidth, coeffs, shift);

    if (ZL_isError(ZL_Output_commit(out, nbElts)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    return ZL_returnSuccess();
}
