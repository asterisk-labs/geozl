#ifndef GEOZL_GPU_PFOR_DECODE_H
#define GEOZL_GPU_PFOR_DECODE_H

#include "frame_batch.h"
#include "pfor_capture.h"
#include "pfor_plan.h"
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

// Buffer indices, in declaration order in gpu/kernels/programs/pfor.slang.
enum {
  PFOR_GPU_OUTPUT = 2,
  PFOR_GPU_OUTPUT_BYTES = 3,
  PFOR_GPU_STATUS = 7,
  PFOR_GPU_BUFFERS = 12
};

typedef struct {
  PforGpuPlan plan;
  PforGpuBandPlan bands;
  uint32_t *status;
  GeozlGpuBuffer buffers[PFOR_GPU_BUFFERS];
  GeozlGpuDispatch dispatches[GEOZL_GPU_MAX_KERNELS];
  size_t dispatchCount;
} PforGpuWork;

int pfor_gpu_status_check(const uint32_t *status, char *error,
                          size_t errorCapacity);

// payload includes GEOZL_GPU_PAYLOAD_PADDING writable bytes.
int pfor_gpu_work_build(PforGpuWork *work, GeozlGpuArena *staging,
                        size_t elementBytes, const PforGpuFrame frames[],
                        size_t frameCount, const size_t frameIds[],
                        uint8_t *payload, size_t payloadSize,
                        const uint8_t *rawValues, size_t rawBytes,
                        const GeozlGpuRawCopy rawCopies[], size_t rawCount,
                        size_t outputBytes, const PforGpuPlan *framePlan,
                        size_t threads, char *error,
                        size_t errorCapacity);
void pfor_gpu_work_free(PforGpuWork *work);

#endif // GEOZL_GPU_PFOR_DECODE_H
