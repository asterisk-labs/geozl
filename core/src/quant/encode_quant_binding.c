#include "encode_quant_binding.h"
#include "encode_quant_kernel.h"
#include "graph_quant.h" // QUANT_GRAPH, QUANT_PARAM_*

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

ZL_Report EI_geozl_quant(ZL_Encoder *eictx, const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  // dtype and the curve parameters are required, the graph builder sets them.
  // The parameters are already resolved against the tile there, because the log
  // curve anchors its grid on the smallest magnitude present and the node has
  // no way to look at the whole raster before the stream reaches it.
  ZL_IntParam dp = ZL_Encoder_getLocalIntParam(eictx, QUANT_PARAM_DTYPE);
  ZL_CopyParam pp = ZL_Encoder_getLocalCopyParam(eictx, QUANT_PARAM_PARAMS);
  if (dp.paramId != QUANT_PARAM_DTYPE || pp.paramId != QUANT_PARAM_PARAMS ||
      pp.paramSize != sizeof(quant_params))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  const int dtype = dp.paramValue;
  quant_params p;
  memcpy(&p, pp.paramPtr, sizeof(p));

  // the index keeps the original element width, so the dtype must name a type
  // of that width
  static const size_t qw[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
  if (dtype < Q_U8 || dtype > Q_F64 || qw[dtype] != eltWidth)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  ZL_Output *out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_ERR_IF_NULL(out, allocation);

  if (quant_encode(ZL_Output_ptr(out), ZL_Input_ptr(in), &p, dtype, nbElts))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  uint8_t header[QUANT_HEADER_SIZE];
  header[0] = (uint8_t)dtype;
  header[1] = p.curve;
  header[2] = p.flags;
  geozl_st_le_f64(header + 3, p.step);
  geozl_st_le_f64(header + 11, p.offset);
  geozl_st_le64(header + 19, p.nsub);
  ZL_Encoder_sendCodecHeader(eictx, header, sizeof(header));

  ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
  return ZL_returnSuccess();
}
