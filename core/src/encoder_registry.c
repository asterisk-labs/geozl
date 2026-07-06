#include "geozl/geozl.h"

#include "average/encode_average_binding.h"
#include "deinterleave/encode_deinterleave_binding.h"
#include "delta_n/encode_delta_n_binding.h"
#include "delta_w/encode_delta_w_binding.h"
#include "med/encode_med_binding.h"
#include "planar/encode_planar_binding.h"
#include "quant_linear/encode_quant_linear_binding.h"
#include "wp_static/encode_wp_static_binding.h"

#include "common/graph_num1to1.h"            // GEOZL_PARAM_WIDTH
#include "quant_linear/graph_quant_linear.h" // QUANT_LINEAR_PARAM_*

#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_localParams.h"

// Register a width predictor and attach the row width, the one param it takes.
static ZL_NodeID width_node(ZL_Compressor* c, const ZL_TypedEncoderDesc* desc,
                            uint32_t width)
{
    ZL_NodeID base = ZL_Compressor_registerTypedEncoder(c, desc);
    if (!ZL_NodeID_isValid(base))
        return base;
    ZL_LocalParams lp    = ZL_LP_1INTPARAM(GEOZL_PARAM_WIDTH, (int)width);
    ZL_NodeParameters np = { .localParams = &lp };
    ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, base, &np);
    return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}

ZL_NodeID geozl_node_delta_w(ZL_Compressor* c, uint32_t width)
{
    return width_node(c, &delta_w_encoder_desc, width);
}
ZL_NodeID geozl_node_delta_n(ZL_Compressor* c, uint32_t width)
{
    return width_node(c, &delta_n_encoder_desc, width);
}
ZL_NodeID geozl_node_planar(ZL_Compressor* c, uint32_t width)
{
    return width_node(c, &planar_encoder_desc, width);
}
ZL_NodeID geozl_node_med(ZL_Compressor* c, uint32_t width)
{
    return width_node(c, &med_encoder_desc, width);
}
ZL_NodeID geozl_node_average(ZL_Compressor* c, uint32_t width)
{
    return width_node(c, &average_encoder_desc, width);
}
ZL_NodeID geozl_node_wp_static(ZL_Compressor* c, uint32_t width)
{
    return width_node(c, &wp_static_encoder_desc, width);
}

ZL_NodeID geozl_node_deinterleave(ZL_Compressor* c)
{
    return ZL_Compressor_registerTypedEncoder(c, &deinterleave_encoder_desc);
}

ZL_NodeID geozl_node_quant_linear(ZL_Compressor* c, double max_error, int dtype)
{
    const double scale = 2.0 * max_error;
    ZL_NodeID base = ZL_Compressor_registerTypedEncoder(c, &quant_linear_encoder_desc);
    if (!ZL_NodeID_isValid(base))
        return base;
    // dtype is an int param, scale a copy param of eight bytes
    const ZL_IntParam ip  = { .paramId    = QUANT_LINEAR_PARAM_DTYPE,
                              .paramValue = dtype };
    const ZL_CopyParam cp = { .paramId  = QUANT_LINEAR_PARAM_SCALE,
                              .paramPtr = &scale,
                              .paramSize = sizeof(scale) };
    ZL_LocalParams lp = {
        .intParams  = { .intParams = &ip, .nbIntParams = 1 },
        .copyParams = { .copyParams = &cp, .nbCopyParams = 1 },
    };
    ZL_NodeParameters np = { .localParams = &lp };
    ZL_RESULT_OF(ZL_NodeID) r = ZL_Compressor_parameterizeNode(c, base, &np);
    return ZL_RES_isError(r) ? ZL_NODE_ILLEGAL : ZL_RES_value(r);
}
