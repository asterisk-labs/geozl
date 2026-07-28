#include "decode_quant_binding.h"
#include "decode_quant_kernel.h"
#include "graph_quant.h" // QUANT_GRAPH, QUANT_HEADER_SIZE

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

ZL_Report DI_geozl_quant(ZL_Decoder *dictx, const ZL_Input *ins[]) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
  assert(ins != NULL);
  const ZL_Input *in = ins[0];
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
  if (header.size != QUANT_HEADER_SIZE)
    return ZL_returnError(ZL_ErrorCode_corruption);
  const uint8_t *hb = (const uint8_t *)header.start;
  const int dtype = (int)hb[0];
  quant_params p;
  memset(&p, 0, sizeof(p));
  p.curve = hb[1];
  p.flags = hb[2];
  p.step = geozl_ld_le_f64(hb + 3);
  p.offset = geozl_ld_le_f64(hb + 11);
  p.nsub = geozl_ld_le64(hb + 19);

  // dtype comes from the header, check it names a real type of the stream
  // width, the same check float_deconstruct makes
  static const size_t qw[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
  if (dtype < Q_U8 || dtype > Q_F64 || qw[dtype] != eltWidth)
    return ZL_returnError(ZL_ErrorCode_corruption);

  // The encoder never writes these, so they can only come from a damaged frame.
  // A stored reconstruction is the linear curve on an integer type alone, the
  // warped reconstructions are not the integer stream the codec emits.
  if (p.curve > QUANT_CURVE_LOG)
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (!isfinite(p.step) || !isfinite(p.offset) || p.step < 0.0)
    return ZL_returnError(ZL_ErrorCode_corruption);
  if ((p.flags & ~(unsigned)QUANT_FLAG_STORE_VALUES) != 0)
    return ZL_returnError(ZL_ErrorCode_corruption);
  if ((p.flags & QUANT_FLAG_STORE_VALUES) != 0 &&
      (p.curve != QUANT_CURVE_LINEAR || dtype > Q_LAST_INT))
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (p.curve == QUANT_CURVE_LOG && !(p.offset > 0.0))
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (p.curve != QUANT_CURVE_LOG && p.nsub != 0)
    return ZL_returnError(ZL_ErrorCode_corruption);

  ZL_Output *out = ZL_Decoder_create1OutStream(dictx, nbElts, eltWidth);
  ZL_ERR_IF_NULL(out, allocation);

  if (quant_decode(ZL_Output_ptr(out), ZL_Input_ptr(in), &p, dtype, nbElts))
    return ZL_returnError(ZL_ErrorCode_corruption);

  ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
  return ZL_returnSuccess();
}