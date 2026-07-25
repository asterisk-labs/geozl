#include "encode_wp_static_binding.h"
#include "encode_wp_static_kernel.h"
#include "train_wp_static.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

ZL_Report EI_geozl_wp_static(ZL_Encoder *eictx, const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  ZL_IntParam wp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_WIDTH);
  const uint32_t width = (wp.paramId == GEOZL_PARAM_WIDTH)
                             ? (uint32_t)wp.paramValue
                             : (uint32_t)nbElts;

  // the weights are fit to this tile, then carried to the decoder in the header
  int16_t coeffs[4];
  uint8_t shift;
  wp_static_train(coeffs, &shift, ZL_Input_ptr(in), width, nbElts, eltWidth);

  ZL_Output *out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_ERR_IF_NULL(out, allocation);

  // nbElts 0 is a valid empty stream, the kernel rejects it as a geometry
  if (nbElts != 0 && wp_static_encode(ZL_Output_ptr(out), ZL_Input_ptr(in),
                                      width, nbElts, eltWidth, coeffs, shift))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  // header, little endian: uint32 width, uint8 shift, four int16
  // {cN,cNW,cNE,cNN}
  uint8_t header[4 + 1 + 4 * 2];
  geozl_st_le32(header, width);
  header[4] = shift;
  for (int i = 0; i < 4; ++i)
    geozl_st_le_i16(header + 5 + 2 * i, coeffs[i]);
  ZL_Encoder_sendCodecHeader(eictx, header, sizeof(header));

  ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
  return ZL_returnSuccess();
}
