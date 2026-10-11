#include "pfor_decode.h"

#include "frame_batch.h"
#include "pfor_capture.h"
#include "pfor_plan.h"
#include "planar_scan.h"
#include "runtime_internal.h"

#include "common/endian.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct GeozlPforGpuDecoder {
  GeozlGpuProgram *program;
  size_t elementBytes;
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

typedef struct {
  uint32_t source;
  uint32_t destination;
  uint32_t count;
} GpuRawCopy;

GeozlPforGpuDecoder *geozl_pfor_gpu_decoder_create(const void *module,
                                                   size_t moduleSize,
                                                   size_t elementBytes,
                                                   int device, char *error,
                                                   size_t errorCapacity) {
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
      module, moduleSize, device, kernels, sizeof(kernels) / sizeof(kernels[0]),
      error, errorCapacity);
  if (decoder->program == NULL) {
    free(decoder);
    return NULL;
  }
  decoder->elementBytes = elementBytes;
  return decoder;
}

void geozl_pfor_gpu_decoder_destroy(GeozlPforGpuDecoder *decoder) {
  if (decoder == NULL)
    return;
  geozl_gpu_program_destroy(decoder->program);
  free(decoder);
}

static int capture_frames(
    const GeozlPforGpuDecoder *decoder, const void *const frames[],
    const size_t frameSizes[], const size_t outputSizes[], size_t frameCount,
    PforGpuFrame gpuFrames[], GeozlGpuFrameShape scanShapes[], uint8_t *payload,
    size_t payloadCapacity, uint8_t *rawValues, GpuRawCopy rawCopies[],
    void *scratch, char *error, size_t errorCapacity, size_t *payloadSize,
    size_t *gpuCount, size_t *scanCount, size_t *rawCount, size_t *rawBytes) {
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
      memcpy(rawValues + *rawBytes, scratch, outputSizes[i]);
      rawCopies[(*rawCount)++] = (GpuRawCopy){
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
    const uint8_t *header = captured.header;
    if ((captured.kind != PFOR_CAPTURE_KIND_ID &&
         captured.kind != PFOR_CAPTURE_KIND_PLANAR_ZIGZAG) ||
        header[8] != decoder->elementBytes) {
      pfor_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu has a different element width", i);
    }
    const uint64_t count64 = geozl_ld_le64(header);
    if (count64 == 0 || count64 > SIZE_MAX ||
        (size_t)count64 > SIZE_MAX / decoder->elementBytes ||
        (size_t)count64 * decoder->elementBytes != outputSizes[i]) {
      pfor_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu has inconsistent output metadata", i);
    }
    GeozlGpuFrameShape shape = {outputElements, (size_t)count64,
                                (size_t)count64, 1};
    int transform = PFOR_GPU_ID;
    if (captured.kind == PFOR_CAPTURE_KIND_PLANAR_ZIGZAG) {
      shape.width = geozl_ld_le32(header + 9);
      shape.planes = geozl_ld_le32(header + 13);
      transform = PFOR_GPU_PLANAR_ZIGZAG;
      scanShapes[(*scanCount)++] = shape;
    }
    gpuFrames[(*gpuCount)++] = (PforGpuFrame){shape, captured.payloadOffset,
                                              captured.payloadSize, transform};
    outputElements += (size_t)count64;
  }
  *payloadSize = capture.payloadSize;
  pfor_capture_destroy(&capture);
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

static int build_dispatches(const PforGpuPlan *plan,
                            const GeozlGpuPlanePlan *planes, size_t rawCount,
                            GeozlGpuDispatch dispatches[], size_t *count,
                            char *error, size_t errorCapacity) {
  *count = 0;
  if (plan->blockCount != 0 &&
      append_dispatch(dispatches, count, K_UNPACK_BLOCKS, plan->blockCount / 4,
                      1, error, errorCapacity))
    return 1;
  if (planes->count != 0 &&
      append_dispatch(dispatches, count, K_SCAN_PLANES, planes->count / 4, 1,
                      error, errorCapacity))
    return 1;
  if (planes->maxWidth > 256 &&
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

  size_t *sizes = calloc(frameCount, sizeof(*sizes));
  PforGpuFrame *gpuFrames = calloc(frameCount, sizeof(*gpuFrames));
  GeozlGpuFrameShape *scanShapes = calloc(frameCount, sizeof(*scanShapes));
  GpuRawCopy *rawCopies = calloc(frameCount, sizeof(*rawCopies));
  if (sizes == NULL || gpuFrames == NULL || scanShapes == NULL ||
      rawCopies == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    goto fail_arrays;
  }

  GeozlGpuFrameBatchInfo batchInfo;
  if (geozl_gpu_frame_batch_inspect(frames, frameSizes, frameCount,
                                    decoder->elementBytes, maxOutputBytes,
                                    sizes, &batchInfo, error, errorCapacity))
    goto fail_arrays;
  uint8_t *payload = malloc(batchInfo.payloadBytes);
  uint8_t *rawValues = malloc(batchInfo.totalBytes);
  void *scratch = malloc(batchInfo.maxFrameBytes);
  if (payload == NULL || rawValues == NULL || scratch == NULL) {
    geozl_gpu_error(error, errorCapacity, "out of memory");
    free(payload);
    free(rawValues);
    free(scratch);
    goto fail_arrays;
  }

  size_t payloadSize = 0;
  size_t gpuCount = 0;
  size_t scanCount = 0;
  size_t rawCount = 0;
  size_t rawBytes = 0;
  if (capture_frames(decoder, frames, frameSizes, sizes, frameCount, gpuFrames,
                     scanShapes, payload, batchInfo.payloadCapacity, rawValues,
                     rawCopies, scratch, error, errorCapacity, &payloadSize,
                     &gpuCount, &scanCount, &rawCount, &rawBytes)) {
    free(payload);
    free(rawValues);
    free(scratch);
    goto fail_arrays;
  }
  free(scratch);
  if (payloadSize > UINT32_MAX - GEOZL_GPU_PAYLOAD_PADDING) {
    geozl_gpu_error(error, errorCapacity, "GPU payload is too large");
    free(payload);
    free(rawValues);
    goto fail_arrays;
  }
  memset(payload + payloadSize, 0, GEOZL_GPU_PAYLOAD_PADDING);

  PforGpuPlan plan = {0};
  if (gpuCount != 0 && pfor_gpu_plan_build_parallel(
                           &plan, gpuFrames, gpuCount, payload, payloadSize,
                           decoder->elementBytes, 4) != 0) {
    if (plan.errorFrame == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s", plan.error);
    else if (plan.errorBlock == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s in frame %zu", plan.error,
                      plan.errorFrame);
    else
      geozl_gpu_error(error, errorCapacity, "%s in block %zu of frame %zu",
                      plan.error, plan.errorBlock, plan.errorFrame);
    free(payload);
    free(rawValues);
    goto fail_arrays;
  }

  GeozlGpuPlanePlan planes = {0};
  if (scanCount != 0 &&
      geozl_gpu_plane_plan_build(&planes, scanShapes, scanCount) != 0) {
    geozl_gpu_error(error, errorCapacity, "invalid planar scan geometry");
    pfor_gpu_plan_free(&plan);
    free(payload);
    free(rawValues);
    goto fail_arrays;
  }

  uint64_t dummy[4] = {0};
  GeozlGpuBuffer buffers[] = {
      {payload, payloadSize + GEOZL_GPU_PAYLOAD_PADDING, 1, GEOZL_GPU_COPY_IN,
       -1},
      {plan.blocks != NULL ? (void *)plan.blocks : (void *)dummy,
       plan.blockCount != 0 ? plan.blockCount * sizeof(*plan.blocks)
                            : sizeof(*plan.blocks),
       sizeof(*plan.blocks), GEOZL_GPU_COPY_IN, -1},
      {planes.tasks != NULL ? (void *)planes.tasks : (void *)dummy,
       planes.count != 0 ? planes.count * sizeof(*planes.tasks)
                         : sizeof(*planes.tasks),
       sizeof(*planes.tasks), GEOZL_GPU_COPY_IN, -1},
      {NULL, batchInfo.totalBytes, decoder->elementBytes, GEOZL_GPU_RESULT, -1},
      {NULL, batchInfo.totalBytes, 1, 0, 3},
      {rawCount != 0 ? rawValues : (uint8_t *)dummy,
       rawCount != 0 ? rawBytes : decoder->elementBytes, decoder->elementBytes,
       GEOZL_GPU_COPY_IN, -1},
      {rawCount != 0 ? rawCopies : (GpuRawCopy *)dummy,
       rawCount != 0 ? rawCount * sizeof(*rawCopies) : sizeof(*rawCopies),
       sizeof(*rawCopies), GEOZL_GPU_COPY_IN, -1},
  };
  GeozlGpuDispatch dispatches[GEOZL_GPU_MAX_KERNELS];
  size_t dispatchCount;
  if (build_dispatches(&plan, &planes, rawCount, dispatches, &dispatchCount,
                       error, errorCapacity))
    goto fail_plan;

  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      decoder->program, buffers, sizeof(buffers) / sizeof(buffers[0]), error,
      errorCapacity);
  if (batch == NULL)
    goto fail_plan;
  GeozlGpuRunTimes times;
  if (geozl_gpu_batch_run(batch, dispatches, dispatchCount, &times, error,
                          errorCapacity) != 0) {
    geozl_gpu_batch_destroy(batch);
    goto fail_plan;
  }
  GeozlGpuMemory *memory = geozl_gpu_batch_take(batch, 3, error, errorCapacity);
  geozl_gpu_batch_destroy(batch);
  if (memory != NULL && outputBytes != NULL)
    *outputBytes = batchInfo.totalBytes;

  geozl_gpu_plane_plan_free(&planes);
  pfor_gpu_plan_free(&plan);
  free(payload);
  free(rawValues);
  free(rawCopies);
  free(scanShapes);
  free(gpuFrames);
  free(sizes);
  return memory;

fail_plan:
  geozl_gpu_plane_plan_free(&planes);
  pfor_gpu_plan_free(&plan);
  free(payload);
  free(rawValues);
fail_arrays:
  free(rawCopies);
  free(scanShapes);
  free(gpuFrames);
  free(sizes);
  return NULL;
}
