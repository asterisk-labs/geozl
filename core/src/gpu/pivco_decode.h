#ifndef GEOZL_GPU_PIVCO_DECODE_H
#define GEOZL_GPU_PIVCO_DECODE_H

#include "frame_batch.h"
#include "pivco_capture.h"
#include "pivco_plan.h"
#include "planar_scan.h"
#include "runtime.h"
#include "staging_arena.h"

#include <stddef.h>

typedef struct GeozlPivcoGpuDecoder GeozlPivcoGpuDecoder;

// module is a Metal library or CUDA PTX compiled from
// gpu/kernels/programs/pivco.slang with the same elementBytes.
GeozlPivcoGpuDecoder *
geozl_pivco_gpu_decoder_create(GeozlGpuBackendId backend, const void *module,
                               size_t moduleSize, size_t elementBytes,
                               int device, char *error, size_t errorCapacity);
void geozl_pivco_gpu_decoder_destroy(GeozlPivcoGpuDecoder *decoder);
GeozlGpuProgram *
geozl_pivco_gpu_decoder_program(const GeozlPivcoGpuDecoder *decoder);

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

// Moves a captured frame's bits payloadBase bytes into the batch payload and
// fills its output shape, starting outputBase elements into the result.
// Returns nonzero when its header disagrees with elementBytes or outputBytes.
int pivco_gpu_frame_init(CapturedFrame *frame, GeozlGpuFrameShape *shape,
                         size_t payloadBase, size_t elementBytes,
                         size_t outputBytes, size_t outputBase);

enum { PIVCO_GPU_OUTPUT = 10, PIVCO_GPU_BUFFERS = 15 };

// Kernel inputs of one batch, built in staging. buffers[PIVCO_GPU_OUTPUT]
// holds the decoded frames, then lane scratch up to resultBytes, and is a
// result the batch allocates unless the caller binds other memory there.
typedef struct {
  PivcoGpuPlan plan;
  GeozlGpuPlanePlan planes;
  GeozlGpuBuffer buffers[PIVCO_GPU_BUFFERS];
  GeozlGpuDispatch dispatches[GEOZL_GPU_MAX_KERNELS];
  size_t dispatchCount;
  size_t resultBytes;
} PivcoGpuWork;

// payload and rawCopies lie in staging, and payload has
// GEOZL_GPU_PAYLOAD_PADDING writable bytes after payloadSize. frameIds, when
// not NULL, names frames in errors.
int pivco_gpu_work_build(PivcoGpuWork *work, GeozlGpuArena *staging,
                         size_t elementBytes, const CapturedFrame frames[],
                         const GeozlGpuFrameShape shapes[], size_t frameCount,
                         const size_t frameIds[], uint8_t *payload,
                         size_t payloadSize, const uint8_t *rawValues,
                         size_t rawBytes, const GeozlGpuRawCopy rawCopies[],
                         size_t rawCount, size_t outputBytes, size_t threads,
                         char *error, size_t errorCapacity);
void pivco_gpu_work_free(PivcoGpuWork *work);

#endif // GEOZL_GPU_PIVCO_DECODE_H
