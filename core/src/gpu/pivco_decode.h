#ifndef GEOZL_GPU_PIVCO_DECODE_H
#define GEOZL_GPU_PIVCO_DECODE_H

#include "runtime.h"

#include <stddef.h>

typedef struct GeozlPivcoGpuDecoder GeozlPivcoGpuDecoder;

// module is a Metal library or CUDA PTX compiled from
// gpu/kernels/programs/pivco.slang with the same elementBytes.
GeozlPivcoGpuDecoder *
geozl_pivco_gpu_decoder_create(GeozlGpuBackendId backend, const void *module,
                               size_t moduleSize, size_t elementBytes,
                               int device, char *error, size_t errorCapacity);
void geozl_pivco_gpu_decoder_destroy(GeozlPivcoGpuDecoder *decoder);

// Decodes frames into one contiguous device allocation, in input order. The
// caller owns the returned memory. No decoded samples are copied to the host.
// maxOutputBytes limits logical decoded bytes; zero disables the caller limit.
// Temporary device storage is additional. The backend's 32-bit byte limit
// remains.
GeozlGpuMemory *geozl_pivco_gpu_decode(GeozlPivcoGpuDecoder *decoder,
                                       const void *const frames[],
                                       const size_t frameSizes[],
                                       size_t frameCount, size_t maxOutputBytes,
                                       size_t *outputBytes, char *error,
                                       size_t errorCapacity);

#endif // GEOZL_GPU_PIVCO_DECODE_H
