#include "decode_quant_linear_binding.h"
#include "decode_quant_linear_kernel.h"
#include "graph_quant_linear.h" // QUANT_LINEAR_GRAPH

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

ZL_Report DI_geozl_quant_linear(ZL_Decoder *dictx, const ZL_Input *ins[]) {
  assert(ins != NULL);
  const ZL_Input *in = ins[0];
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  // header, little endian: uint8 dtype, then the scale as an IEEE double
  ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
  if (header.size != 1 + sizeof(double))
    return ZL_returnError(ZL_ErrorCode_corruption);
  const uint8_t *hb = (const uint8_t *)header.start;
  const int dtype = (int)hb[0];
  double scale;
  memcpy(&scale, hb + 1, sizeof(scale));

  // dtype comes from the header, check it names a real type of the stream
  // width, the same check float_deconstruct makes
  static const size_t qlw[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
  if (dtype < QL_U8 || dtype > QL_F64 || qlw[dtype] != eltWidth)
    return ZL_returnError(ZL_ErrorCode_corruption);

  // The encoder never writes inf or nan. A negative scale means the stream
  // already holds reconstructed values and only integers can carry those, the
  // float reconstruction does not share the index type.
  if (!isfinite(scale))
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (scale < 0.0 && dtype > QL_I64)
    return ZL_returnError(ZL_ErrorCode_corruption);

  ZL_Output *out = ZL_Decoder_create1OutStream(dictx, nbElts, eltWidth);
  if (out == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  quant_linear_decode(ZL_Output_ptr(out), ZL_Input_ptr(in), scale, dtype,
                      nbElts);

  if (ZL_isError(ZL_Output_commit(out, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}