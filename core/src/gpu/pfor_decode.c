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
  K_SCAN_PLANES,
  K_SCAN_ROWS,
  K_SCAN_COLUMNS,
  K_COPY_RAW,
};

static const char *const kernels[] = {
    "unpack_blocks", "scan_planes", "scan_rows", "scan_columns", "copy_raw",
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
  const GeozlGpuPlanePlan *planes = &work->planes;
  GeozlGpuDispatch *dispatches = work->dispatches;
  size_t *count = &work->dispatchCount;
  *count = 0;
  if (plan->blockCount != 0 &&
      append_dispatch(dispatches, count, K_UNPACK_BLOCKS, plan->blockCount / 4,
                      1, error, errorCapacity))
    return 1;
  if (planes->count != 0 &&
      append_dispatch(dispatches, count, K_SCAN_PLANES, planes->count / 4, 1,
                      error, errorCapacity))
    return 1;
  if (planes->maxWidth > GEOZL_GPU_SCAN_SEGMENT &&
      (append_dispatch(dispatches, count, K_SCAN_ROWS, planes->count,
                       (planes->maxRows + 3) / 4, error, errorCapacity) ||
       append_dispatch(dispatches, count, K_SCAN_COLUMNS, planes->count,
                       (planes->maxWidth + 127) / 128, error, errorCapacity)))
    return 1;
  if (rawCount != 0 && append_dispatch(dispatches, count, K_COPY_RAW, rawCount,
                                       1, error, errorCapacity))
    return 1;
  return 0;
}

static size_t frame_id(const size_t frameIds[], size_t frame) {
  return frameIds == NULL ? frame : frameIds[frame];
}

int pfor_gpu_work_build(PforGpuWork *work, GeozlGpuArena *staging,
                        size_t elementBytes, const PforGpuFrame frames[],
                        size_t frameCount, const size_t frameIds[],
                        uint8_t *payload, size_t payloadSize,
                        const uint8_t *rawValues, size_t rawBytes,
                        const GeozlGpuRawCopy rawCopies[], size_t rawCount,
                        size_t outputBytes, size_t threads, char *error,
                        size_t errorCapacity) {
  *work = (PforGpuWork){0};
  if (payloadSize > UINT32_MAX - GEOZL_GPU_PAYLOAD_PADDING)
    return geozl_gpu_error(error, errorCapacity, "GPU payload is too large");
  if (outputBytes == 0 || outputBytes > UINT32_MAX ||
      outputBytes % elementBytes != 0)
    return geozl_gpu_error(error, errorCapacity, "GPU output is too large");
  const GeozlGpuHostAlloc alloc = geozl_gpu_arena_allocator(staging);
  uint64_t *dummy = geozl_gpu_host_calloc(&alloc, 4, sizeof(*dummy));
  if (dummy == NULL)
    return geozl_gpu_error(error, errorCapacity, "%s",
                           geozl_gpu_arena_error(staging));
  memset(payload + payloadSize, 0, GEOZL_GPU_PAYLOAD_PADDING);

  if (frameCount != 0 && pfor_gpu_plan_build_parallel(
                             &work->plan, frames, frameCount, payload,
                             payloadSize, elementBytes, threads, &alloc) != 0) {
    const PforGpuPlan *plan = &work->plan;
    if (plan->errorFrame == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s", plan->error);
    else if (plan->errorBlock == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s in frame %zu", plan->error,
                      frame_id(frameIds, plan->errorFrame));
    else
      geozl_gpu_error(error, errorCapacity, "%s in block %zu of frame %zu",
                      plan->error, plan->errorBlock,
                      frame_id(frameIds, plan->errorFrame));
    return 1;
  }

  GeozlGpuFrameShape *scans =
      frameCount == 0 ? NULL : malloc(frameCount * sizeof(*scans));
  if (frameCount != 0 && scans == NULL) {
    pfor_gpu_work_free(work);
    return geozl_gpu_error(error, errorCapacity, "out of memory");
  }
  size_t scanCount = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    if (frames[i].transform == PFOR_GPU_PLANAR_ZIGZAG)
      scans[scanCount++] = frames[i].shape;
  }
  const int planned =
      scanCount == 0 ||
      geozl_gpu_plane_plan_build(&work->planes, scans, scanCount, &alloc) == 0;
  free(scans);
  if (!planned) {
    pfor_gpu_work_free(work);
    return geozl_gpu_error(error, errorCapacity,
                           "invalid planar scan geometry");
  }

  const PforGpuPlan *plan = &work->plan;
  const GeozlGpuPlanePlan *planes = &work->planes;
  GeozlGpuBuffer *buffers = work->buffers;
  buffers[0] = geozl_gpu_arena_input(
      staging, payload, payloadSize + GEOZL_GPU_PAYLOAD_PADDING, 1);
  buffers[1] = geozl_gpu_arena_input(
      staging, plan->blocks != NULL ? (void *)plan->blocks : dummy,
      plan->blockCount, sizeof(*plan->blocks));
  buffers[2] = geozl_gpu_arena_input(
      staging, planes->tasks != NULL ? (void *)planes->tasks : dummy,
      planes->count, sizeof(*planes->tasks));
  buffers[PFOR_GPU_OUTPUT] = (GeozlGpuBuffer){.bytes = outputBytes,
                                              .elementBytes = elementBytes,
                                              .flags = GEOZL_GPU_RESULT,
                                              .alias = -1};
  buffers[4] = (GeozlGpuBuffer){
      .bytes = outputBytes, .elementBytes = 1, .alias = PFOR_GPU_OUTPUT};
  buffers[5] = geozl_gpu_arena_input(
      staging, rawValues != NULL ? (const void *)rawValues : dummy,
      rawBytes / elementBytes, elementBytes);
  buffers[6] = geozl_gpu_arena_input(
      staging, rawCount != 0 ? (const void *)rawCopies : dummy, rawCount,
      sizeof(*rawCopies));
  if (build_dispatches(work, rawCount, error, errorCapacity)) {
    pfor_gpu_work_free(work);
    return 1;
  }
  return 0;
}

void pfor_gpu_work_free(PforGpuWork *work) {
  geozl_gpu_plane_plan_free(&work->planes);
  pfor_gpu_plan_free(&work->plan);
}

// Raw frames are rare, so their staging is taken at the first one, sized for
// the output still to come.
static int capture_frames(GeozlPforGpuDecoder *decoder,
                          const void *const frames[], const size_t frameSizes[],
                          const size_t outputSizes[], size_t frameCount,
                          size_t totalBytes, PforGpuFrame gpuFrames[],
                          uint8_t *payload, size_t payloadCapacity,
                          uint8_t **rawValues, GeozlGpuRawCopy rawCopies[],
                          void *scratch, char *error, size_t errorCapacity,
                          size_t *payloadSize, size_t *gpuCount,
                          size_t *rawCount, size_t *rawBytes) {
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
    ++*gpuCount;
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

  // Kernel inputs are built in staging, which the device reads in place.
  GeozlGpuArena *staging = &decoder->staging;
  geozl_gpu_arena_reset(staging);
  const GeozlGpuHostAlloc alloc = geozl_gpu_arena_allocator(staging);
  PforGpuWork work = {0};
  GeozlGpuMemory *memory = NULL;
  void *scratch = NULL;
  size_t *sizes = calloc(frameCount, sizeof(*sizes));
  PforGpuFrame *gpuFrames = calloc(frameCount, sizeof(*gpuFrames));
  if (sizes == NULL || gpuFrames == NULL) {
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
                     batchInfo.totalBytes, gpuFrames, payload,
                     batchInfo.payloadCapacity, &rawValues, rawCopies, scratch,
                     error, errorCapacity, &payloadSize, &gpuCount, &rawCount,
                     &rawBytes) ||
      pfor_gpu_work_build(&work, staging, decoder->elementBytes, gpuFrames,
                          gpuCount, NULL, payload, payloadSize, rawValues,
                          rawBytes, rawCopies, rawCount, batchInfo.totalBytes,
                          4, error, errorCapacity))
    goto done;

  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      decoder->program, work.buffers, PFOR_GPU_BUFFERS, error, errorCapacity);
  if (batch == NULL)
    goto done;
  GeozlGpuRunTimes times;
  if (geozl_gpu_batch_run(batch, work.dispatches, work.dispatchCount, &times,
                          error, errorCapacity) == 0)
    memory = geozl_gpu_batch_take(batch, PFOR_GPU_OUTPUT, error, errorCapacity);
  geozl_gpu_batch_destroy(batch);
  if (memory != NULL && outputBytes != NULL)
    *outputBytes = batchInfo.totalBytes;

done:
  pfor_gpu_work_free(&work);
  free(scratch);
  free(gpuFrames);
  free(sizes);
  return memory;
}
