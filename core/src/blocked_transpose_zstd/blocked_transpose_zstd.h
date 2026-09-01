#ifndef GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_H
#define GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_H

#include "geozl/export.h"

#include <stddef.h>

// Raw helpers shared by the Python and C custom transforms.
GEOZL_API size_t geozl_blocked_transpose_zstd_bound(size_t nbElts,
                                                    size_t eltWidth,
                                                    size_t blockSize);
GEOZL_API int geozl_blocked_transpose_zstd_encode(
    void *dst, size_t dstCapacity, size_t *outSize, const void *src,
    size_t nbElts, size_t eltWidth, size_t blockSize, int compressionLevel);
GEOZL_API int geozl_blocked_transpose_zstd_decode(
    void *dst, size_t nbElts, size_t eltWidth, size_t blockSize,
    const void *src, size_t srcSize);

#endif // GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_H
