#include "encode_floatquant_binding.h"
#include "encode_floatquant_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>

#define GEOZL_PARAM_FLOATQUANT_K 1

ZL_Report EI_geozl_floatquant(ZL_Encoder *eictx, const ZL_Input *in) {
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (eltWidth != 4 && eltWidth != 8)
    return ZL_returnError(ZL_ErrorCode_GENERIC);

  ZL_IntParam kp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_FLOATQUANT_K);
  const unsigned precBits = (eltWidth == 4) ? 23u : 52u;
  if (kp.paramId != GEOZL_PARAM_FLOATQUANT_K || kp.paramValue < 1 ||
      (unsigned)kp.paramValue > precBits)
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  const unsigned k = (unsigned)kp.paramValue;

  ZL_Output *s0 = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_Output *s1 = ZL_Encoder_createTypedStream(eictx, 1, nbElts, eltWidth);
  if (s0 == NULL || s1 == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  floatquant_split(ZL_Output_ptr(s0), ZL_Output_ptr(s1), ZL_Input_ptr(in),
                   nbElts, eltWidth, k);

  const uint8_t header[1] = {(uint8_t)k};
  ZL_Encoder_sendCodecHeader(eictx, header, 1);

  if (ZL_isError(ZL_Output_commit(s0, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  if (ZL_isError(ZL_Output_commit(s1, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}
