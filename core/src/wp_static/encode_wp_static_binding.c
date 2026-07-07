#include "encode_wp_static_binding.h"
#include "encode_wp_static_kernel.h"
#include "train_wp_static.h"

#include "common/graph_num1to1.h" // GEOZL_NUM1TO1_GRAPH, GEOZL_PARAM_WIDTH
#include "common/raster.h"    // geozl_row_width
#include "geozl/ctids.h"          // GEOZL_CTID_WP_STATIC

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

const ZL_TypedEncoderDesc wp_static_encoder_desc = {
    .gd          = GEOZL_NUM1TO1_GRAPH(GEOZL_CTID_WP_STATIC),
    .transform_f = EI_geozl_wp_static,
    .name        = "geozl.lossless.wp_static",
};

ZL_Report EI_geozl_wp_static(ZL_Encoder* eictx, const ZL_Input* in)
{
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_numeric);

    const size_t eltWidth = ZL_Input_eltWidth(in);
    const size_t nbElts   = ZL_Input_numElts(in);

    ZL_IntParam wp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_WIDTH);
    const uint32_t width = (wp.paramId == GEOZL_PARAM_WIDTH)
            ? (uint32_t)wp.paramValue
            : (uint32_t)nbElts;

    if (nbElts != 0 && geozl_row_width(width, nbElts) == 0)
        return ZL_returnError(ZL_ErrorCode_node_invalid_input);

    // the weights are fit to this tile, then carried to the decoder in the header
    int16_t coeffs[4];
    uint8_t shift;
    wp_static_train(coeffs, &shift, ZL_Input_ptr(in), width, nbElts, eltWidth);

    ZL_Output* out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
    if (out == NULL)
        return ZL_returnError(ZL_ErrorCode_allocation);

    wp_static_encode(ZL_Output_ptr(out), ZL_Input_ptr(in), width, nbElts,
                     eltWidth, coeffs, shift);

    // header, little endian: uint32 width, uint8 shift, four int16 {cN,cNW,cNE,cNN}
    uint8_t header[4 + 1 + 4 * sizeof(int16_t)];
    memcpy(header, &width, sizeof(width));
    header[4] = shift;
    memcpy(header + 5, coeffs, 4 * sizeof(int16_t));
    ZL_Encoder_sendCodecHeader(eictx, header, sizeof(header));

    if (ZL_isError(ZL_Output_commit(out, nbElts)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    return ZL_returnSuccess();
}
