#include "encode_intmult_binding.h"
#include "encode_intmult_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// base comes from a node parameter (Python detector, or a caller that knows the
// grid). This C encode path is not compiled today; kept for a future C
// detector. Param id mirrors the Python side.
#define GEOZL_PARAM_INTMULT_BASE 1

ZL_Report EI_geozl_intmult(ZL_Encoder *eictx, const ZL_Input *in) {
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8)
    return ZL_returnError(ZL_ErrorCode_GENERIC);

  ZL_IntParam bp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_INTMULT_BASE);
  if (bp.paramId != GEOZL_PARAM_INTMULT_BASE || bp.paramValue < 2)
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  const uint64_t base = (uint64_t)bp.paramValue;

  ZL_Output *s0 = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_Output *s1 = ZL_Encoder_createTypedStream(eictx, 1, nbElts, eltWidth);
  if (s0 == NULL || s1 == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  intmult_split(ZL_Output_ptr(s0), ZL_Output_ptr(s1), ZL_Input_ptr(in), nbElts,
                eltWidth, base);

  uint8_t header[8];
  memcpy(header, &base, eltWidth); // little endian
  ZL_Encoder_sendCodecHeader(eictx, header, eltWidth);

  if (ZL_isError(ZL_Output_commit(s0, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  if (ZL_isError(ZL_Output_commit(s1, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}
