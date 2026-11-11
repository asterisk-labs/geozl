#include "pfor_decode.h"

#include "runtime_internal.h"

#include "common/endian.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct GeozlPforGpuDecoder {
  GeozlGpuProgram *program;
  size_t elementBytes;
  GeozlGpuArena staging;
};

enum {
  K_UNPACK_BLOCKS,
  K_SIZE_CHUNKS,
  K_PLAN_BLOCKS,
  K_DECODE_STRIPS,
  K_COPY_RAW,
};

static const char *const kernels[] = {
    "unpack_blocks",
    "size_chunks",
    "plan_blocks",
    "decode_strips",
    "copy_raw",
};

GeozlPforGpuDecoder *
geozl_pfor_gpu_decoder_create(GeozlGpuBackendId backend, const void *module,
                              size_t moduleSize, size_t elementBytes,
                              int device, char *error, size_t errorCapacity) {
  if (elementBytes != 1 && elementBytes != 2) {
    geozl_gpu_error(error, errorCapacity,
                    "PFOR GPU decoding supports 1- or 2-byte elements");
    return NULL;
  }
  GeozlPforGpuDecoder *decoder = calloc(1, sizeof(*decoder));
  if (decoder == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    return NULL;
  }
  decoder->program = geozl_gpu_program_create(
      backend, module, moduleSize, device, kernels,
      sizeof(kernels) / sizeof(kernels[0]), error, errorCapacity);
  if (decoder->program == NULL) {
    free(decoder);
    return NULL;
  }
  decoder->elementBytes = elementBytes;
  geozl_gpu_arena_init(&decoder->staging, decoder->program);
  return decoder;
}

void geozl_pfor_gpu_decoder_destroy(GeozlPforGpuDecoder *decoder) {
  if (decoder == NULL)
    return;
  geozl_gpu_arena_destroy(&decoder->staging);
  geozl_gpu_program_destroy(decoder->program);
  free(decoder);
}

GeozlGpuProgram *
geozl_pfor_gpu_decoder_program(const GeozlPforGpuDecoder *decoder) {
  return decoder == NULL ? NULL : decoder->program;
}

int pfor_gpu_frame_init(PforGpuFrame *frame, const CapturedPforFrame *captured,
                        size_t payloadBase, size_t elementBytes,
                        size_t outputBytes, size_t outputBase) {
  const uint8_t *header = captured->header;
  if ((captured->kind != PFOR_CAPTURE_KIND_ID &&
       captured->kind != PFOR_CAPTURE_KIND_PLANAR_ZIGZAG) ||
      header[8] != elementBytes)
    return 1;
  const uint64_t count = geozl_ld_le64(header);
  if (count == 0 || count > SIZE_MAX / elementBytes ||
      (size_t)count * elementBytes != outputBytes ||
      captured->payloadOffset > SIZE_MAX - payloadBase)
    return 1;
  *frame = (PforGpuFrame){
      .shape = {outputBase, (size_t)count, (size_t)count, 1},
      .payloadOffset = payloadBase + captured->payloadOffset,
      .payloadSize = captured->payloadSize,
      .transform = PFOR_GPU_ID,
  };
  if (captured->kind == PFOR_CAPTURE_KIND_PLANAR_ZIGZAG) {
    frame->shape.width = geozl_ld_le32(header + 9);
    frame->shape.planes = geozl_ld_le32(header + 13);
    frame->transform = PFOR_GPU_PLANAR_ZIGZAG;
  }
  return 0;
}

static int append_dispatch(GeozlGpuDispatch dispatches[], size_t *count,
                           unsigned kernel, size_t groupsX, size_t groupsY,
                           char *error, size_t errorCapacity) {
  if (groupsX == 0 || groupsY == 0 || groupsX > UINT_MAX ||
      groupsY > UINT_MAX || *count >= GEOZL_GPU_MAX_KERNELS)
    return geozl_gpu_error(error, errorCapacity,
                           "GPU dispatch dimensions exceed their limits");
  dispatches[(*count)++] =
      (GeozlGpuDispatch){kernel, (unsigned)groupsX, (unsigned)groupsY, 128};
  return 0;
}

static int build_dispatches(PforGpuWork *work, size_t rawCount, char *error,
                            size_t errorCapacity) {
  const PforGpuPlan *plan = &work->plan;
  const PforGpuBandPlan *bands = &work->bands;
  GeozlGpuDispatch *dispatches = work->dispatches;
  size_t *count = &work->dispatchCount;
  *count = 0;
  if (plan->blockCount != 0 &&
      append_dispatch(dispatches, count, K_UNPACK_BLOCKS, plan->blockCount / 4,
                      1, error, errorCapacity))
    return 1;
  if (bands->frameCount != 0 &&
      (append_dispatch(dispatches, count, K_SIZE_CHUNKS, bands->chunkCount / 4,
                       1, error, errorCapacity) ||
       append_dispatch(dispatches, count, K_PLAN_BLOCKS, bands->chunkCount / 4,
                       1, error, errorCapacity) ||
       append_dispatch(dispatches, count, K_DECODE_STRIPS,
                       bands->stripCount / 4, 1, error, errorCapacity)))
    return 1;
  if (rawCount != 0 && append_dispatch(dispatches, count, K_COPY_RAW, rawCount,
                                       1, error, errorCapacity))
    return 1;
  return 0;
}

static size_t frame_id(const size_t frameIds[], size_t frame) {
  return frameIds == NULL ? frame : frameIds[frame];
}

int pfor_gpu_status_check(const uint32_t *status, char *error,
                          size_t errorCapacity) {
  if (status == NULL || (status[PFOR_GPU_STATUS_HEADER] == UINT32_MAX &&
                         status[PFOR_GPU_STATUS_POSITIONS] == UINT32_MAX))
    return 0;
  const int header =
      status[PFOR_GPU_STATUS_HEADER] <= status[PFOR_GPU_STATUS_POSITIONS];
  return geozl_gpu_error(error, errorCapacity, "frame %u: %s",
                         header ? status[PFOR_GPU_STATUS_HEADER]
                                : status[PFOR_GPU_STATUS_POSITIONS],
                         header ? "invalid PFOR block header or payload size"
                                : "invalid PFOR exception positions");
}

int pfor_gpu_work_build(PforGpuWork *work, GeozlGpuArena *staging,
                        size_t elementBytes, const PforGpuFrame frames[],
                        size_t frameCount, const size_t frameIds[],
                        uint8_t *payload, size_t payloadSize,
                        const uint8_t *rawValues, size_t rawBytes,
                        const GeozlGpuRawCopy rawCopies[], size_t rawCount,
                        size_t outputBytes, const PforGpuPlan *framePlan,
                        size_t threads, char *error,
                        size_t errorCapacity) {
  *work = (PforGpuWork){0};
  if (payloadSize > UINT32_MAX - GEOZL_GPU_PAYLOAD_PADDING)
    return geozl_gpu_error(error, errorCapacity, "GPU payload is too large");
  if (outputBytes == 0 || outputBytes > UINT32_MAX ||
      outputBytes % elementBytes != 0)
    return geozl_gpu_error(error, errorCapacity, "GPU output is too large");
  const GeozlGpuHostAlloc alloc = geozl_gpu_arena_allocator(staging);
  uint64_t *dummy = geozl_gpu_host_calloc(&alloc, 4, sizeof(*dummy));
  work->status = geozl_gpu_host_calloc(&alloc, 4, sizeof(*work->status));
  if (dummy == NULL || work->status == NULL)
    return geozl_gpu_error(error, errorCapacity, "%s",
                           geozl_gpu_arena_error(staging));
  work->status[PFOR_GPU_STATUS_HEADER] = UINT32_MAX;
  work->status[PFOR_GPU_STATUS_POSITIONS] = UINT32_MAX;
  memset(payload + payloadSize, 0, GEOZL_GPU_PAYLOAD_PADDING);

  size_t idCount = 0;
  for (size_t i = 0; i < frameCount; ++i)
    idCount += frames[i].transform == PFOR_GPU_ID;
  PforGpuFrame *idFrames = NULL;
  size_t *idIds = NULL;
  if (framePlan == NULL && idCount != 0 && idCount != frameCount) {
    idFrames = malloc(idCount * sizeof(*idFrames));
    idIds = malloc(idCount * sizeof(*idIds));
    if (idFrames == NULL || idIds == NULL) {
      free(idFrames);
      free(idIds);
      return geozl_gpu_error(error, errorCapacity, "out of memory");
    }
    size_t k = 0;
    for (size_t i = 0; i < frameCount; ++i) {
      if (frames[i].transform != PFOR_GPU_ID)
        continue;
      idFrames[k] = frames[i];
      idIds[k++] = frame_id(frameIds, i);
    }
  }
  const PforGpuFrame *planFrames = idFrames != NULL ? idFrames : frames;
  const size_t *planIds = idFrames != NULL ? idIds : frameIds;
  int planFailed = 0;
  if (framePlan != NULL) {
    work->plan = *framePlan;
  } else if (idCount != 0 &&
             pfor_gpu_plan_build_parallel(&work->plan, planFrames, idCount,
                                          payload, payloadSize, elementBytes,
                                          threads, &alloc) != 0) {
    const PforGpuPlan *plan = &work->plan;
    if (plan->errorFrame == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s", plan->error);
    else if (plan->errorBlock == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s in frame %zu", plan->error,
                      frame_id(planIds, plan->errorFrame));
    else
      geozl_gpu_error(error, errorCapacity, "%s in block %zu of frame %zu",
                      plan->error, plan->errorBlock,
                      frame_id(planIds, plan->errorFrame));
    planFailed = 1;
  }
  free(idFrames);
  free(idIds);
  if (planFailed)
    return 1;

  if (pfor_gpu_band_plan_build(&work->bands, frames, frameCount, frameIds,
                               &alloc) != 0) {
    const PforGpuBandPlan *bands = &work->bands;
    if (bands->errorFrame == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s", bands->error);
    else
      geozl_gpu_error(error, errorCapacity, "%s in frame %zu", bands->error,
                      frame_id(frameIds, bands->errorFrame));
    pfor_gpu_work_free(work);
    return 1;
  }

  const PforGpuPlan *plan = &work->plan;
  const PforGpuBandPlan *bands = &work->bands;
  GeozlGpuBuffer *buffers = work->buffers;
  buffers[0] = geozl_gpu_arena_input(
      staging, payload, payloadSize + GEOZL_GPU_PAYLOAD_PADDING, 1);
  buffers[1] = geozl_gpu_arena_input(
      staging, plan->blocks != NULL ? (void *)plan->blocks : dummy,
      plan->blockCount, sizeof(*plan->blocks));
  buffers[PFOR_GPU_OUTPUT] = (GeozlGpuBuffer){.bytes = outputBytes,
                                              .elementBytes = elementBytes,
                                              .flags = GEOZL_GPU_RESULT,
                                              .alias = -1};
  buffers[PFOR_GPU_OUTPUT_BYTES] = (GeozlGpuBuffer){
      .bytes = outputBytes, .elementBytes = 1, .alias = PFOR_GPU_OUTPUT};
  buffers[4] = geozl_gpu_arena_input(
      staging, bands->frames != NULL ? (void *)bands->frames : dummy,
      bands->frameCount, sizeof(*bands->frames));
  buffers[5] = geozl_gpu_arena_input(
      staging, bands->strips != NULL ? (void *)bands->strips : dummy,
      bands->stripCount, sizeof(*bands->strips));
  buffers[6] = (GeozlGpuBuffer){
      .bytes = (bands->blockCount == 0 ? 1 : bands->blockCount) *
               sizeof(GpuPforBlockRef),
      .elementBytes = sizeof(GpuPforBlockRef),
      .alias = -1};
  buffers[PFOR_GPU_STATUS] =
      geozl_gpu_arena_input(staging, work->status, 2, sizeof(*work->status));
  buffers[PFOR_GPU_STATUS].flags |= GEOZL_GPU_COPY_OUT;
  buffers[8] = geozl_gpu_arena_input(
      staging, bands->chunks != NULL ? (void *)bands->chunks : dummy,
      bands->chunkCount, sizeof(*bands->chunks));
  buffers[9] = (GeozlGpuBuffer){
      .bytes = (bands->chunkCount == 0 ? 1 : bands->chunkCount) *
               sizeof(uint32_t),
      .elementBytes = sizeof(uint32_t),
      .alias = -1};
  buffers[10] = geozl_gpu_arena_input(
      staging, rawValues != NULL ? (const void *)rawValues : dummy,
      rawBytes / elementBytes, elementBytes);
  buffers[11] = geozl_gpu_arena_input(
      staging, rawCount != 0 ? (const void *)rawCopies : dummy, rawCount,
      sizeof(*rawCopies));
  if (build_dispatches(work, rawCount, error, errorCapacity)) {
    pfor_gpu_work_free(work);
    return 1;
  }
  return 0;
}

void pfor_gpu_work_free(PforGpuWork *work) {
  pfor_gpu_band_plan_free(&work->bands);
  pfor_gpu_plan_free(&work->plan);
}

static int
capture_frames(GeozlPforGpuDecoder *decoder, const void *const frames[],
               const size_t frameSizes[], const size_t outputSizes[],
               size_t frameCount, size_t totalBytes, PforGpuFrame gpuFrames[],
               size_t gpuIds[], uint8_t *payload, size_t payloadCapacity,
               uint8_t **rawValues, GeozlGpuRawCopy rawCopies[], void *scratch,
               char *error, size_t errorCapacity, size_t *payloadSize,
               size_t *gpuCount, size_t *rawCount, size_t *rawBytes) {
  PforCapture capture;
  if (pfor_capture_init(&capture, payload, payloadCapacity) != 0)
    return geozl_gpu_error(error, errorCapacity,
                           "cannot create the PFOR frame parser");

  size_t outputElements = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    CapturedPforFrame captured = {0};
    const PforCaptureResult result = pfor_capture_frame(
        &capture, &captured, scratch, outputSizes[i], frames[i], frameSizes[i]);
    if (result == PFOR_CAPTURE_RAW) {
      if (captured.rawElementBytes != decoder->elementBytes ||
          *rawBytes > UINT32_MAX || outputElements > UINT32_MAX ||
          outputSizes[i] / decoder->elementBytes > UINT32_MAX) {
        pfor_capture_destroy(&capture);
        return geozl_gpu_error(error, errorCapacity,
                               "frame %zu has a different element width", i);
      }
      if (*rawValues == NULL &&
          (*rawValues = geozl_gpu_arena_alloc(
               &decoder->staging,
               totalBytes - outputElements * decoder->elementBytes)) == NULL) {
        pfor_capture_destroy(&capture);
        return geozl_gpu_error(error, errorCapacity, "%s",
                               geozl_gpu_arena_error(&decoder->staging));
      }
      memcpy(*rawValues + *rawBytes, scratch, outputSizes[i]);
      rawCopies[(*rawCount)++] = (GeozlGpuRawCopy){
          (uint32_t)(*rawBytes / decoder->elementBytes),
          (uint32_t)outputElements,
          (uint32_t)(outputSizes[i] / decoder->elementBytes),
      };
      *rawBytes += outputSizes[i];
      outputElements += outputSizes[i] / decoder->elementBytes;
      continue;
    }
    if (result != PFOR_CAPTURE_CODEC) {
      pfor_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu does not end in a supported PFOR codec",
                             i);
    }
    if (pfor_gpu_frame_init(&gpuFrames[*gpuCount], &captured, 0,
                            decoder->elementBytes, outputSizes[i],
                            outputElements) != 0) {
      pfor_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu has inconsistent output metadata", i);
    }
    gpuIds[(*gpuCount)++] = i;
    outputElements += outputSizes[i] / decoder->elementBytes;
  }
  *payloadSize = capture.payloadSize;
  pfor_capture_destroy(&capture);
  return 0;
}

GeozlGpuMemory *geozl_pfor_gpu_decode(GeozlPforGpuDecoder *decoder,
                                      const void *const frames[],
                                      const size_t frameSizes[],
                                      size_t frameCount, size_t maxOutputBytes,
                                      size_t *outputBytes, char *error,
                                      size_t errorCapacity) {
  if (error != NULL && errorCapacity != 0)
    error[0] = '\0';
  if (outputBytes != NULL)
    *outputBytes = 0;
  if (decoder == NULL || frames == NULL || frameSizes == NULL ||
      frameCount == 0 || frameCount > UINT32_MAX) {
    geozl_gpu_error(error, errorCapacity, "invalid PFOR GPU decode input");
    return NULL;
  }

  GeozlGpuArena *staging = &decoder->staging;
  geozl_gpu_arena_reset(staging);
  const GeozlGpuHostAlloc alloc = geozl_gpu_arena_allocator(staging);
  PforGpuWork work = {0};
  GeozlGpuMemory *memory = NULL;
  void *scratch = NULL;
  size_t *sizes = calloc(frameCount, sizeof(*sizes));
  PforGpuFrame *gpuFrames = calloc(frameCount, sizeof(*gpuFrames));
  size_t *gpuIds = calloc(frameCount, sizeof(*gpuIds));
  if (sizes == NULL || gpuFrames == NULL || gpuIds == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    goto done;
  }

  GeozlGpuFrameBatchInfo batchInfo;
  if (geozl_gpu_frame_batch_inspect(frames, frameSizes, frameCount,
                                    decoder->elementBytes, maxOutputBytes,
                                    sizes, &batchInfo, error, errorCapacity))
    goto done;
  GeozlGpuRawCopy *rawCopies =
      geozl_gpu_host_calloc(&alloc, frameCount, sizeof(*rawCopies));
  uint8_t *payload = geozl_gpu_arena_alloc(staging, batchInfo.payloadBytes);
  if (rawCopies == NULL || payload == NULL) {
    geozl_gpu_error(error, errorCapacity, "%s", geozl_gpu_arena_error(staging));
    goto done;
  }
  scratch = malloc(batchInfo.maxFrameBytes);
  if (scratch == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    goto done;
  }

  uint8_t *rawValues = NULL;
  size_t payloadSize = 0;
  size_t gpuCount = 0;
  size_t rawCount = 0;
  size_t rawBytes = 0;
  if (capture_frames(decoder, frames, frameSizes, sizes, frameCount,
                     batchInfo.totalBytes, gpuFrames, gpuIds, payload,
                     batchInfo.payloadCapacity, &rawValues, rawCopies, scratch,
                     error, errorCapacity, &payloadSize, &gpuCount, &rawCount,
                     &rawBytes) ||
      pfor_gpu_work_build(&work, staging, decoder->elementBytes, gpuFrames,
                          gpuCount, gpuIds, payload, payloadSize, rawValues,
                          rawBytes, rawCopies, rawCount, batchInfo.totalBytes,
                          NULL, 4, error, errorCapacity))
    goto done;

  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      decoder->program, work.buffers, PFOR_GPU_BUFFERS, error, errorCapacity);
  if (batch == NULL)
    goto done;
  GeozlGpuRunTimes times;
  if (geozl_gpu_batch_run(batch, work.dispatches, work.dispatchCount, &times,
                          error, errorCapacity) == 0 &&
      pfor_gpu_status_check(work.status, error, errorCapacity) == 0)
    memory = geozl_gpu_batch_take(batch, PFOR_GPU_OUTPUT, error, errorCapacity);
  geozl_gpu_batch_destroy(batch);
  if (memory != NULL && outputBytes != NULL)
    *outputBytes = batchInfo.totalBytes;

done:
  pfor_gpu_work_free(&work);
  free(scratch);
  free(gpuIds);
  free(gpuFrames);
  free(sizes);
  return memory;
}
