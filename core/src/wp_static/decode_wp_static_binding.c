#include "decode_wp_static_binding.h"
#include "decode_wp_static_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// Codec header, little endian: uint32 width, uint8 shift, then WP_STATIC_NTAPS
// int16 coefficients in the order N, NW, NE, NN.
#define WP_STATIC_HEADER_SIZE (4 + 1 + WP_STATIC_NTAPS * 2)

ZL_Report DI_geozl_wp_static(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    assert(ins != NULL);
    const ZL_Input* in = ins[0];
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_numeric);

    const size_t eltWidth = ZL_Input_eltWidth(in);
    const size_t nbElts   = ZL_Input_numElts(in);

    ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
    if (header.size != WP_STATIC_HEADER_SIZE)
        return ZL_returnError(ZL_ErrorCode_corruption);

    const uint8_t* h = (const uint8_t*)header.start;
    uint32_t width;
    memcpy(&width, h, sizeof(width));
    const uint8_t shift = h[4];
    int16_t coeffs[WP_STATIC_NTAPS];
    memcpy(coeffs, h + 5, sizeof(coeffs));

    ZL_Output* out = ZL_Decoder_create1OutStream(dictx, nbElts, eltWidth);
    if (out == NULL)
        return ZL_returnError(ZL_ErrorCode_allocation);

    wp_static_decode(ZL_Output_ptr(out), ZL_Input_ptr(in), width, nbElts,
                     eltWidth, coeffs, shift);

    if (ZL_isError(ZL_Output_commit(out, nbElts)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    return ZL_returnSuccess();
}
