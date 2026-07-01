#include "geozl/geozl.h"

#include "openzl/zl_errors.h"

ZL_Report geozl_register_decoders(ZL_DCtx* dctx)
{
    ZL_Report r = geozl_register_lossless_decoders(dctx);
    if (ZL_isError(r)) return r;
    return geozl_register_lossy_decoders(dctx);
}