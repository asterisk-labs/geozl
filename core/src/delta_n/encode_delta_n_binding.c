#include "encode_delta_n_binding.h"
#include "encode_delta_n_kernel.h"

#include "common/graph_num1to1.h" // GEOZL_NUM1TO1_GRAPH, GEOZL_PARAM_WIDTH
#include "geozl/ctids.h"          // GEOZL_CTID_DELTA_N

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>

const ZL_TypedEncoderDesc delta_n_encoder_desc = {
    .gd          = GEOZL_NUM1TO1_GRAPH(GEOZL_CTID_DELTA_N),
    .transform_f = EI_geozl_delta_n,
    .name        = "geozl.lossless.delta_n",
};

ZL_Report EI_geozl_delta_n(ZL_Encoder* eictx, const ZL_Input* in)
{
    // guaranteed by the engine and the codec signature
    assert(in != NULL);
    assert(ZL_Input_type(in) == ZL_Type_numeric);

    const size_t eltWidth = ZL_Input_eltWidth(in);
    const size_t nbElts   = ZL_Input_numElts(in);

    // the row width comes from the graph builder, a single row if it is absent
    ZL_IntParam wp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_WIDTH);
    const uint32_t width = (wp.paramId == GEOZL_PARAM_WIDTH)
            ? (uint32_t)wp.paramValue
            : (uint32_t)nbElts;

    // allocation is controlled by the engine
    ZL_Output* out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
    if (out == NULL)
        return ZL_returnError(ZL_ErrorCode_allocation);

    delta_n_encode(ZL_Output_ptr(out), ZL_Input_ptr(in), width, nbElts, eltWidth);

    // the width is all the decoder needs, it rides in the codec header
    ZL_Encoder_sendCodecHeader(eictx, &width, sizeof(width));

    if (ZL_isError(ZL_Output_commit(out, nbElts)))
        return ZL_returnError(ZL_ErrorCode_GENERIC);
    return ZL_returnSuccess();
}
