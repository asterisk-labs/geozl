#ifndef GEOZL_H
#define GEOZL_H

#include "openzl/zl_dtransform.h"
#include "geozl/export.h"
#include "geozl/kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register the geozl custom decoders into dctx. A consumer that decodes geozl
// frames must register both, since a frame can carry either family. The
// decoders are stateless, one set per dctx per thread is fine.
GEOZL_API ZL_Report geozl_register_lossless_decoders(ZL_DCtx* dctx);
GEOZL_API ZL_Report geozl_register_lossy_decoders(ZL_DCtx* dctx);

// Both families in one call.
GEOZL_API ZL_Report geozl_register_decoders(ZL_DCtx* dctx);

#ifdef __cplusplus
}
#endif

#endif