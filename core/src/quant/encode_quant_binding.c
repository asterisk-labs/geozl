#include "encode_quant_binding.h"
#include "graph_quant.h" // QUANT_PARAM_*, QUANT_HEADER_SIZE
#include "quant_dtype.h" // Q_U8, Q_F64
#include "quant_spec.h"  // quant_fit

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

#include "common/endian.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ZL_Report EI_geozl_quant(ZL_Encoder *eictx, const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);

  // All three are required. The curve parameters arrive already resolved
  // against the tile, since the log curve anchors on the smallest magnitude in
  // the raster and this node only ever sees one stream.
  ZL_IntParam dp = ZL_Encoder_getLocalIntParam(eictx, QUANT_PARAM_DTYPE);
  ZL_CopyParam pp = ZL_Encoder_getLocalCopyParam(eictx, QUANT_PARAM_PARAMS);
  ZL_CopyParam sp2 = ZL_Encoder_getLocalCopyParam(eictx, QUANT_PARAM_SPEC);
  if (dp.paramId != QUANT_PARAM_DTYPE || pp.paramId != QUANT_PARAM_PARAMS ||
      pp.paramSize != sizeof(quant_params) || sp2.paramId != QUANT_PARAM_SPEC ||
      sp2.paramSize != sizeof(quant_spec))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  const int dtype = dp.paramValue;
  quant_params p;
  quant_spec sp;
  memcpy(&p, pp.paramPtr, sizeof(p));
  memcpy(&sp, sp2.paramPtr, sizeof(sp));

  // the index keeps the original element width, so the dtype must name a type
  // of that width
  if (dtype < Q_U8 || dtype > Q_F64 || quant_width(dtype) != eltWidth)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  ZL_Output *out = ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
  ZL_ERR_IF_NULL(out, allocation);

  // Measure the round trip against the bound the recipe declared and tighten
  // until it holds, rather than trusting that the parameters were derived
  // right. p comes back carrying the step the frame was actually written with.
  void *chk = malloc(nbElts * eltWidth);
  ZL_ERR_IF_NULL(chk, allocation);
  const int fit =
      quant_fit(ZL_Output_ptr(out), chk, ZL_Input_ptr(in), &sp, &p, dtype,
                nbElts);
  free(chk);
  // A tile no grid can serve is a recipe the caller has to loosen. Kernels that
  // reject parameters the resolver just produced is neither, and reporting the
  // two the same way sends that one to the wrong person.
  if (fit < 0)
    ZL_ERR(GENERIC, "quant resolved parameters its own kernels reject");
  if (fit != 0)
    ZL_ERR(GENERIC, "quant cannot hold its declared error on this tile");

  uint8_t header[QUANT_HEADER_SIZE];
  header[0] = (uint8_t)dtype;
  header[1] = p.curve;
  header[2] = p.flags;
  geozl_st_le_f64(header + 3, p.step);
  geozl_st_le_f64(header + 11, p.offset);
  geozl_st_le64(header + 19, p.nsub);
  ZL_Encoder_sendCodecHeader(eictx, header, sizeof(header));

  ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
  return ZL_returnSuccess();
}