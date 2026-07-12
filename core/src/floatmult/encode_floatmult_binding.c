#include "encode_floatmult_binding.h"
#include "encode_floatmult_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// base travels as an f64 bit pattern in two int params (hi/lo 32 bits), since
// local params are integers. The Python encoder is the primary user.
#define GEOZL_PARAM_FLOATMULT_BASE_LO 1
#define GEOZL_PARAM_FLOATMULT_BASE_HI 2

ZL_Report EI_geozl_floatmult(ZL_Encoder *eictx, const ZL_Input *in) {
  assert(in != NULL);
  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (eltWidth != 4 && eltWidth != 8)
    return ZL_returnError(ZL_ErrorCode_GENERIC);

  ZL_IntParam lo =
      ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_FLOATMULT_BASE_LO);
  ZL_IntParam hi =
      ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_FLOATMULT_BASE_HI);
  if (lo.paramId != GEOZL_PARAM_FLOATMULT_BASE_LO ||
      hi.paramId != GEOZL_PARAM_FLOATMULT_BASE_HI)
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  uint64_t baseBits = ((uint64_t)(uint32_t)hi.paramValue << 32) |
                      (uint64_t)(uint32_t)lo.paramValue;
  double base;
  memcpy(&base, &baseBits, sizeof(double));

  ZL_Output *s0 = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_Output *s1 = ZL_Encoder_createTypedStream(eictx, 1, nbElts, eltWidth);
  if (s0 == NULL || s1 == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);
  floatmult_split(ZL_Output_ptr(s0), ZL_Output_ptr(s1), ZL_Input_ptr(in),
                  nbElts, eltWidth, base, 1.0 / base);
  ZL_Encoder_sendCodecHeader(eictx, &baseBits, sizeof(double));
  if (ZL_isError(ZL_Output_commit(s0, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  if (ZL_isError(ZL_Output_commit(s1, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}
