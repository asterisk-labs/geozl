#include "decode_average_binding.h"
#include "decode_average_kernel.h"

#include "common/raster.h" // geozl_row_width

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

ZL_Report DI_geozl_average(ZL_Decoder *dictx, const ZL_Input *ins[]) {
  assert(ins != NULL);
  const ZL_Input *in = ins[0];
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  // the width is carried in the codec header, written by the encoder
  ZL_RBuffer header = ZL_Decoder_getCodecHeader(dictx);
  if (header.size != sizeof(uint32_t))
    return ZL_returnError(ZL_ErrorCode_corruption);
  uint32_t width;
  memcpy(&width, header.start, sizeof(width));

  if (nbElts != 0 && geozl_row_width(width, nbElts) == 0)
    return ZL_returnError(ZL_ErrorCode_corruption);

  ZL_Output *out = ZL_Decoder_create1OutStream(dictx, nbElts, eltWidth);
  if (out == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  average_decode(ZL_Output_ptr(out), ZL_Input_ptr(in), width, nbElts, eltWidth);

  if (ZL_isError(ZL_Output_commit(out, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}
