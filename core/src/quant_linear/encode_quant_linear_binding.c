#include "encode_quant_linear_binding.h"
#include "encode_quant_linear_kernel.h"
#include "graph_quant_linear.h" // QUANT_LINEAR_GRAPH, QUANT_LINEAR_PARAM_*

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

ZL_Report EI_geozl_quant_linear(ZL_Encoder *eictx, const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  // dtype and scale are required, the graph builder sets them. scale is a
  // double so it comes through a copy param, an int param would truncate it.
  ZL_IntParam dp = ZL_Encoder_getLocalIntParam(eictx, QUANT_LINEAR_PARAM_DTYPE);
  ZL_CopyParam sp =
      ZL_Encoder_getLocalCopyParam(eictx, QUANT_LINEAR_PARAM_SCALE);
  if (dp.paramId != QUANT_LINEAR_PARAM_DTYPE ||
      sp.paramId != QUANT_LINEAR_PARAM_SCALE || sp.paramSize != sizeof(double))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  const int dtype = dp.paramValue;
  double scale;
  memcpy(&scale, sp.paramPtr, sizeof(scale));

  // the index keeps the original element width, so the dtype must name a type
  // of that width
  static const size_t qlw[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
  if (dtype < QL_U8 || dtype > QL_F64 || qlw[dtype] != eltWidth)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  ZL_Output *out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_ERR_IF_NULL(out, allocation);

  if (quant_linear_encode(ZL_Output_ptr(out), ZL_Input_ptr(in), scale, dtype,
                          nbElts))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  // header, little endian: uint8 dtype, then the scale as an IEEE double
  uint8_t header[1 + 8];
  header[0] = (uint8_t)dtype;
  geozl_st_le_f64(header + 1, scale);
  ZL_Encoder_sendCodecHeader(eictx, header, sizeof(header));

  ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
  return ZL_returnSuccess();
}
