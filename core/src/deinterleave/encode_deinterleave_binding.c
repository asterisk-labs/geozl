#include "encode_deinterleave_binding.h"
#include "encode_deinterleave_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>

ZL_Report EI_geozl_deinterleave(ZL_Encoder *eictx, const ZL_Input *in) {
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  const size_t n0 = (nbElts + 1) / 2; // even lane
  const size_t n1 = nbElts / 2;       // odd lane

  ZL_Output *s0 = ZL_Encoder_createTypedStream(eictx, 0, n0, eltWidth);
  ZL_Output *s1 = ZL_Encoder_createTypedStream(eictx, 1, n1, eltWidth);
  if (s0 == NULL || s1 == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  deinterleave_split(ZL_Output_ptr(s0), ZL_Output_ptr(s1), ZL_Input_ptr(in),
                     nbElts, eltWidth);

  // no codec header, the decoder rebuilds the order from the two lane counts
  if (ZL_isError(ZL_Output_commit(s0, n0)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  if (ZL_isError(ZL_Output_commit(s1, n1)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}
