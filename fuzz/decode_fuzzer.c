// Feeds arbitrary bytes to the geozl decode path. Checksums are off so mutated
// frames reach the decoders instead of being rejected first. A crash means a
// decoder trusts a header field it should have checked.

#include "geozl/geozl.h"

#include "openzl/zl_common_types.h" // ZL_TernaryParam_disable
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static const size_t kMaxOut = 64u << 20; // cap the allocation so a forged size cannot OOM

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ZL_DCtx* dctx = ZL_DCtx_create();
    if (dctx == NULL)
        return 0;
    (void)geozl_register_decoders(dctx);
    (void)ZL_DCtx_setParameter(dctx, ZL_DParam_checkCompressedChecksum, ZL_TernaryParam_disable);
    (void)ZL_DCtx_setParameter(dctx, ZL_DParam_checkContentChecksum, ZL_TernaryParam_disable);

    const ZL_Report want = ZL_getDecompressedSize(data, size);
    if (!ZL_isError(want)) {
        const size_t cap = ZL_validResult(want);
        if (cap <= kMaxOut) {
            void* dst = malloc(cap ? cap : 1);
            if (dst != NULL) {
                (void)ZL_DCtx_decompress(dctx, dst, cap, data, size);
                free(dst);
            }
        }
    }

    ZL_DCtx_free(dctx);
    return 0;
}