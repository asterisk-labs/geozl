#include "decode_wp_static_binding.h"
#include "decode_wp_static_kernel.h"

#include "common/graph_num1to1.h" // GEOZL_NUM1TO1_GRAPH
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

const ZL_TypedDecoderDesc wp_static_decoder_desc = {
    .gd          = GEOZL_NUM1TO1_GRAPH(GEOZL_CTID_WP_STATIC),
    .transform_f = DI_geozl_wp_static,
    .name        = "geozl.lossless.wp_static",
};

ZL_Report DI_geozl_wp_static(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    assert(ins != NULL);
    const ZL_Input* in = ins[0];
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_numeric);

    const size_t eltWidth = ZL_Input_eltWidth(in);
    const size_t nbElts   = ZL_Input_numElts(in);

    // header, little endian: uint32 width, uint8 shift, four int16 {cN,cNW,cNE,cNN}
    ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
    if (header.size != 4 + 1 + 4 * sizeof(int16_t))
        return ZL_returnError(ZL_ErrorCode_corruption);
    const uint8_t* hb = (const uint8_t*)header.start;
    uint32_t width;
    int16_t coeffs[4];
    uint8_t shift;
    memcpy(&width, hb, sizeof(width));
    shift = hb[4];
    memcpy(coeffs, hb + 5, 4 * sizeof(int16_t));

    if (nbElts != 0 && geozl_row_width(width, nbElts) == 0)
        return ZL_returnError(ZL_ErrorCode_corruption);

    // the kernel folds the sum in 64-bit, so a shift of 64 or more is undefined
    if (shift >= 64)
        return ZL_returnError(ZL_ErrorCode_corruption);

    ZL_Output* out = ZL_Decoder_create1OutStream(dictx, nbElts, eltWidth);
    if (out == NULL)
        return ZL_returnError(ZL_ErrorCode_allocation);

    wp_static_decode(ZL_Output_ptr(out), ZL_Input_ptr(in), width, nbElts,
                     eltWidth, coeffs, shift);

    if (ZL_isError(ZL_Output_commit(out, nbElts)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    return ZL_returnSuccess();
}