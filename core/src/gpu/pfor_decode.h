#ifndef GEOZL_GPU_PFOR_DECODE_H
#define GEOZL_GPU_PFOR_DECODE_H

#include "runtime.h"

#include <stddef.h>

typedef struct GeozlPforGpuDecoder GeozlPforGpuDecoder;

GeozlPforGpuDecoder *
geozl_pfor_gpu_decoder_create(GeozlGpuBackendId backend, const void *module,
                              size_t moduleSize, size_t elementBytes,
                              int device, char *error, size_t errorCapacity);
void geozl_pfor_gpu_decoder_destroy(GeozlPforGpuDecoder *decoder);

// Decodes frames into one contiguous device allocation, in input order. The
// caller owns the returned memory. maxOutputBytes limits logical decoded bytes;
// zero disables the caller limit. Temporary device storage is additional. The
// backend's 32-bit byte limit remains.
GeozlGpuMemory *geozl_pfor_gpu_decode(GeozlPforGpuDecoder *decoder,
                                      const void *const frames[],
                                      const size_t frameSizes[],
                                      size_t frameCount, size_t maxOutputBytes,
                                      size_t *outputBytes, char *error,
                                      size_t errorCapacity);

#endif // GEOZL_GPU_PFOR_DECODE_H
