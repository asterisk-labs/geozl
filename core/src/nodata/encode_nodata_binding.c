#include "encode_nodata_binding.h"
#include "encode_nodata_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <stdint.h>

ZL_Report EI_geozl_nodata(ZL_Encoder *eictx, const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  // An empty tile has no mask to carry and the decoder refuses one, so writing
  // it would make a frame nothing can read.
  if (nbElts == 0)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  ZL_IntParam mp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_NODATA_PARAM_MODE);
  if (mp.paramId != GEOZL_NODATA_PARAM_MODE)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  const int mode = mp.paramValue;

  ZL_IntParam wp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_NODATA_PARAM_WIDTH);
  const uint32_t width = (wp.paramId == GEOZL_NODATA_PARAM_WIDTH)
                             ? (uint32_t)wp.paramValue
                             : (uint32_t)nbElts;

  // Output 0 keeps the sample width, output 1 is one byte of mask per sample.
  ZL_Output *vals = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_Output *mask = ZL_Encoder_createTypedStream(eictx, 1, nbElts, 1);
  ZL_ERR_IF_NULL(vals, allocation);
  ZL_ERR_IF_NULL(mask, allocation);

  uint64_t pattern = 0;
  uint8_t *mp8 = (uint8_t *)ZL_Output_ptr(mask);
  if (mode == GEOZL_NODATA_NAN) {
    // The marking is the NaN test itself, so a second payload is a hole too,
    // and the pattern is the first one found.
    nodata_find_nan(&pattern, ZL_Input_ptr(in), nbElts, eltWidth);
    nodata_mark_nan(mp8, ZL_Input_ptr(in), nbElts, eltWidth);
  } else if (mode == GEOZL_NODATA_VALUE) {
    ZL_CopyParam vp =
        ZL_Encoder_getLocalCopyParam(eictx, GEOZL_NODATA_PARAM_VALUE);
    if (vp.paramId != GEOZL_NODATA_PARAM_VALUE || vp.paramSize != 8)
      return ZL_returnError(ZL_ErrorCode_node_invalid_input);
    pattern = geozl_ld_le64((const uint8_t *)vp.paramPtr);
    nodata_mark_value(mp8, ZL_Input_ptr(in), nbElts, eltWidth, pattern);
  } else {
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  }

  // One shape on the wire, so the header is the pattern and the count is the
  // length of a stream OpenZL already sized.
  uint8_t header[8];
  geozl_st_le(header, pattern, eltWidth);
  ZL_Encoder_sendCodecHeader(eictx, header, eltWidth);

  nodata_fill(ZL_Output_ptr(vals), ZL_Input_ptr(in),
              (const uint8_t *)ZL_Output_ptr(mask), width, nbElts, eltWidth);

  ZL_ERR_IF_ERR(ZL_Output_commit(vals, nbElts));
  ZL_ERR_IF_ERR(ZL_Output_commit(mask, nbElts));
  return ZL_returnSuccess();
}
