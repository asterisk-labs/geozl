#include "decode_wp_static_binding.h"
#include "decode_wp_static_kernel.h"
#include "header_wp_static.h"

#include "openzl/zl_data.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>


ZL_Report DI_geozl_wp_static(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    assert(ins != NULL);
    const ZL_Input* in = ins[0];
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_numeric);

    const size_t eltWidth = ZL_Input_eltWidth(in);
    const size_t nbElts   = ZL_Input_numElts(in);

    ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
    uint32_t width;
    int16_t coeffs[4];
    uint8_t shift;
    if (!wp_static_read_header((const uint8_t*)header.start, header.size,
                               &width, coeffs, &shift))
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
