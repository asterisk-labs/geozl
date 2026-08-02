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
#include <string.h>

ZL_Report EI_geozl_nodata(ZL_Encoder *eictx, const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8)
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
  // Both are sized for the whole tile and committed short when the code below
  // drops one of them.
  ZL_Output *vals = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_Output *mask = ZL_Encoder_createTypedStream(eictx, 1, nbElts, 1);
  ZL_ERR_IF_NULL(vals, allocation);
  ZL_ERR_IF_NULL(mask, allocation);

  uint64_t pattern = 0;
  size_t holes = 0;
  if (nbElts != 0) {
    uint8_t *mp8 = (uint8_t *)ZL_Output_ptr(mask);
    if (mode == GEOZL_NODATA_MODE_NAN) {
      // The pattern is what gets restored, the marking is the NaN test itself,
      // so a second payload is a hole too. A tile with no NaN leaves the
      // pattern at 0, the mask all valid, and the codec a pass through.
      nodata_find_nan(&pattern, ZL_Input_ptr(in), nbElts, eltWidth);
      holes = nodata_mark_nan(mp8, ZL_Input_ptr(in), nbElts, eltWidth);
    } else if (mode == GEOZL_NODATA_MODE_VALUE) {
      ZL_CopyParam vp =
          ZL_Encoder_getLocalCopyParam(eictx, GEOZL_NODATA_PARAM_VALUE);
      if (vp.paramId != GEOZL_NODATA_PARAM_VALUE || vp.paramSize != 8)
        return ZL_returnError(ZL_ErrorCode_node_invalid_input);
      pattern = geozl_ld_le64((const uint8_t *)vp.paramPtr);
      holes = nodata_mark_value(mp8, ZL_Input_ptr(in), nbElts, eltWidth,
                                pattern);
    } else {
      return ZL_returnError(ZL_ErrorCode_node_invalid_input);
    }
  }

  // header, little endian: uint8 code, then whatever that code needs
  uint8_t header[1 + 8 + 8];
  size_t headerSize;
  size_t nbVals = nbElts, nbMask = nbElts;

  if (holes == 0) {
    // No fill has anything to cover, so the values are the input untouched.
    header[0] = GEOZL_NODATA_WIRE_ALL_VALID;
    headerSize = 1;
    nbMask = 0;
    if (nbElts != 0)
      memcpy(ZL_Output_ptr(vals), ZL_Input_ptr(in), nbElts * eltWidth);
  } else if (holes == nbElts) {
    header[0] = GEOZL_NODATA_WIRE_ALL_HOLE;
    geozl_st_le(header + 1, pattern, eltWidth);
    geozl_st_le64(header + 1 + eltWidth, (uint64_t)nbElts);
    headerSize = 1 + eltWidth + 8;
    nbVals = 0;
    nbMask = 0;
  } else {
    header[0] = GEOZL_NODATA_WIRE_RESTORE;
    geozl_st_le(header + 1, pattern, eltWidth);
    headerSize = 1 + eltWidth;
    nodata_fill(ZL_Output_ptr(vals), ZL_Input_ptr(in),
                (const uint8_t *)ZL_Output_ptr(mask), width, nbElts, eltWidth);
  }
  ZL_Encoder_sendCodecHeader(eictx, header, headerSize);

  ZL_ERR_IF_ERR(ZL_Output_commit(vals, nbVals));
  ZL_ERR_IF_ERR(ZL_Output_commit(mask, nbMask));
  return ZL_returnSuccess();
}
