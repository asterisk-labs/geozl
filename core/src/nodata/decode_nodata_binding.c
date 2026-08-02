#include "decode_nodata_binding.h"
#include "decode_nodata_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

ZL_Report DI_geozl_nodata(ZL_Decoder *dictx, const ZL_Input *ins[]) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
  assert(ins != NULL);
  const ZL_Input *vals = ins[0];
  const ZL_Input *mask = ins[1];
  assert(vals != NULL && mask != NULL);
  assert(ZL_Input_type(vals) == ZL_Type_numeric);
  assert(ZL_Input_type(mask) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(vals);
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8)
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (ZL_Input_eltWidth(mask) != 1)
    return ZL_returnError(ZL_ErrorCode_corruption);

  // header, little endian, uint8 code then whatever that code needs. The
  // stream sizes are what checks the code is telling the truth.
  ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
  if (header.size == 0)
    return ZL_returnError(ZL_ErrorCode_corruption);
  const uint8_t *hb = (const uint8_t *)header.start;

  size_t nbElts;
  uint64_t pattern = 0;
  switch (hb[0]) {
  case GEOZL_NODATA_WIRE_ALL_VALID:
    if (header.size != 1 || ZL_Input_numElts(mask) != 0)
      return ZL_returnError(ZL_ErrorCode_corruption);
    nbElts = ZL_Input_numElts(vals);
    break;
  case GEOZL_NODATA_WIRE_RESTORE:
    if (header.size != 1 + eltWidth)
      return ZL_returnError(ZL_ErrorCode_corruption);
    nbElts = ZL_Input_numElts(vals);
    if (ZL_Input_numElts(mask) != nbElts || nbElts == 0)
      return ZL_returnError(ZL_ErrorCode_corruption);
    pattern = geozl_ld_le(hb + 1, eltWidth);
    break;
  case GEOZL_NODATA_WIRE_ALL_HOLE: {
    if (header.size != 1 + eltWidth + 8 || ZL_Input_numElts(vals) != 0 ||
        ZL_Input_numElts(mask) != 0)
      return ZL_returnError(ZL_ErrorCode_corruption);
    pattern = geozl_ld_le(hb + 1, eltWidth);
    // The count sizes an allocation and comes from the frame, so it has to be
    // caught before it is multiplied out.
    const uint64_t stored = geozl_ld_le64(hb + 1 + eltWidth);
    if (stored == 0 || stored > (uint64_t)(SIZE_MAX / eltWidth))
      return ZL_returnError(ZL_ErrorCode_corruption);
    nbElts = (size_t)stored;
    break;
  }
  default:
    return ZL_returnError(ZL_ErrorCode_corruption);
  }

  ZL_Output *out = ZL_Decoder_create1OutStream(dictx, nbElts, eltWidth);
  ZL_ERR_IF_NULL(out, allocation);

  if (hb[0] == GEOZL_NODATA_WIRE_ALL_HOLE)
    nodata_broadcast(ZL_Output_ptr(out), nbElts, eltWidth, pattern);
  else if (hb[0] == GEOZL_NODATA_WIRE_ALL_VALID)
    memcpy(ZL_Output_ptr(out), ZL_Input_ptr(vals), nbElts * eltWidth);
  else
    nodata_restore(ZL_Output_ptr(out), ZL_Input_ptr(vals), ZL_Input_ptr(mask),
                   nbElts, eltWidth, pattern);

  ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
  return ZL_returnSuccess();
}
