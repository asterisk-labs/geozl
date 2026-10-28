#ifndef GEOZL_GPU_PFOR_DECODE_H
#define GEOZL_GPU_PFOR_DECODE_H

#include "frame_batch.h"
#include "pfor_capture.h"
#include "pfor_plan.h"
#include "planar_scan.h"
#include "runtime.h"
#include "staging_arena.h"

#include <stddef.h>

typedef struct GeozlPforGpuDecoder GeozlPforGpuDecoder;

GeozlPforGpuDecoder *
geozl_pfor_gpu_decoder_create(GeozlGpuBackendId backend, const void *module,
                              size_t moduleSize, size_t elementBytes,
                              int device, char *error, size_t errorCapacity);
void geozl_pfor_gpu_decoder_destroy(GeozlPforGpuDecoder *decoder);
GeozlGpuProgram *
geozl_pfor_gpu_decoder_program(const GeozlPforGpuDecoder *decoder);

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

// Plans a captured frame whose payload starts payloadBase bytes into the batch
// payload and whose output starts outputBase elements into the result.
// Returns nonzero when its header disagrees with elementBytes or outputBytes.
int pfor_gpu_frame_init(PforGpuFrame *frame, const CapturedPforFrame *captured,
                        size_t payloadBase, size_t elementBytes,
                        size_t outputBytes, size_t outputBase);

enum { PFOR_GPU_OUTPUT = 3, PFOR_GPU_BUFFERS = 7 };

// Kernel inputs of one batch, built in staging. buffers[PFOR_GPU_OUTPUT] holds
// the decoded frames and is a result the batch allocates unless the caller
// binds other memory there.
typedef struct {
  PforGpuPlan plan;
  GeozlGpuPlanePlan planes;
  GeozlGpuBuffer buffers[PFOR_GPU_BUFFERS];
  GeozlGpuDispatch dispatches[GEOZL_GPU_MAX_KERNELS];
  size_t dispatchCount;
} PforGpuWork;

// payload and rawCopies lie in staging, and payload has
// GEOZL_GPU_PAYLOAD_PADDING writable bytes after payloadSize. frameIds, when
// not NULL, names frames in errors.
int pfor_gpu_work_build(PforGpuWork *work, GeozlGpuArena *staging,
                        size_t elementBytes, const PforGpuFrame frames[],
                        size_t frameCount, const size_t frameIds[],
                        uint8_t *payload, size_t payloadSize,
                        const uint8_t *rawValues, size_t rawBytes,
                        const GeozlGpuRawCopy rawCopies[], size_t rawCount,
                        size_t outputBytes, size_t threads, char *error,
                        size_t errorCapacity);
void pfor_gpu_work_free(PforGpuWork *work);

#endif // GEOZL_GPU_PFOR_DECODE_H
