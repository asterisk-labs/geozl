#ifndef GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_ENCODE_KERNEL_H
#define GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_ENCODE_KERNEL_H

#include <stddef.h>

// Shuffle an array of fixed-width values into contiguous byte lanes. Lanes are
// ordered least-significant byte first, independent of host endianness.
void blocked_transpose_zstd_shuffle(void *dst, const void *src, size_t nbElts,
                                    size_t eltWidth);

#endif // GEOZL_CODECS_BLOCKED_TRANSPOSE_ZSTD_ENCODE_KERNEL_H
