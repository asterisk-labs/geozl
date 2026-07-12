#include "encode_binoffset_binding.h"
#include "encode_binoffset_kernel.h"
#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// C encoder: emit the implicit power-of-two table (bin = bit_length) as an
// explicit header, so the decoder is always table-driven. The Python encoder
// writes the same layout with optimizer-built tables.
ZL_Report EI_geozl_binoffset(ZL_Encoder *eictx, const ZL_Input *in) {
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (eltWidth != 1 && eltWidth != 2 && eltWidth != 4 && eltWidth != 8)
    return ZL_returnError(ZL_ErrorCode_GENERIC);

  ZL_Output *s0 = ZL_Encoder_createTypedStream(eictx, 0, nbElts, 1);
  ZL_Output *s1 = ZL_Encoder_createTypedStream(eictx, 1, nbElts, eltWidth);
  if (s0 == NULL || s1 == NULL)
    return ZL_returnError(ZL_ErrorCode_allocation);

  binoffset_split(ZL_Output_ptr(s0), ZL_Output_ptr(s1), ZL_Input_ptr(in),
                  nbElts, eltWidth);

  // bin b covers [2^(b-1), 2^b); bin 0 holds only 0
  const unsigned nbBins = (unsigned)(8u * eltWidth) + 1u;
  uint8_t header[1 + 65 * 9];
  uint8_t *e = header;
  *e++ = (uint8_t)nbBins;
  for (unsigned b = 0; b < nbBins; ++b) {
    const uint64_t lower = b ? (uint64_t)1 << (b - 1u) : 0u;
    memcpy(e, &lower, eltWidth); // little endian
    e[eltWidth] = (uint8_t)(b ? b - 1u : 0u);
    e += eltWidth + 1;
  }
  ZL_Encoder_sendCodecHeader(eictx, header, (size_t)(e - header));

  if (ZL_isError(ZL_Output_commit(s0, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  if (ZL_isError(ZL_Output_commit(s1, nbElts)))
    return ZL_returnError(ZL_ErrorCode_GENERIC);
  return ZL_returnSuccess();
}
