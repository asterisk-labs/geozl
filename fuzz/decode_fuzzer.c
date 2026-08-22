#include "geozl/geozl.h"

#include "openzl/zl_common_types.h" // ZL_TernaryParam_disable
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static const size_t kMaxOut = 64u << 20; // a forged size field must not OOM

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Exercise coefficient extraction in both copy and size-query modes.
  {
    static uint8_t blob[16384];
    size_t got = 0;
    (void)geozl_2d_frame_coeffs_c(data, size, blob, sizeof blob, &got);
    (void)geozl_2d_frame_coeffs_c(data, size, NULL, 0, &got);
  }

  ZL_DCtx *dctx = ZL_DCtx_create();
  if (dctx == NULL)
    return 0;
  (void)geozl_register_decoders(dctx);
  (void)ZL_DCtx_setParameter(dctx, ZL_DParam_checkCompressedChecksum,
                             ZL_TernaryParam_disable);
  (void)ZL_DCtx_setParameter(dctx, ZL_DParam_checkContentChecksum,
                             ZL_TernaryParam_disable);

  const ZL_Report want = ZL_getDecompressedSize(data, size);
  if (!ZL_isError(want)) {
    const size_t cap = ZL_validResult(want);
    if (cap <= kMaxOut) {
      // malloc is max-aligned, which the numeric output needs.
      void *dst = malloc(cap ? cap : 1);
      if (dst != NULL) {
        // A geozl frame carries one numeric output, so the typed call is the one
        // that reaches the graph. The serial call turns the frame away on type
        // before a decoder runs.
        ZL_OutputInfo info;
        (void)ZL_DCtx_decompressTyped(dctx, &info, dst, cap, data, size);
        free(dst);
      }
    }
  }

  ZL_DCtx_free(dctx);
  return 0;
}
