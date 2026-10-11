#include "pivco_decode.h"

#include "frame_batch.h"
#include "pivco_capture.h"
#include "pivco_plan.h"
#include "planar_scan.h"
#include "runtime_internal.h"

#include "common/endian.h"

#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct GeozlPivcoGpuDecoder {
  GeozlGpuProgram *program;
  size_t elementBytes;
};

enum {
  K_DECODE_SYMBOLS,
  K_UNPACK_LANE_BLOCKS,
  K_JOIN_LANES,
  K_SCAN_PLANES,
  K_SCAN_ROWS,
  K_SCAN_COLUMNS,
  K_COPY_RAW,
};

static const char *const kernels[] = {
    "decode_symbols", "unpack_lane_blocks", "join_lanes", "scan_planes",
    "scan_rows",      "scan_columns",       "copy_raw",
};

typedef struct {
  uint32_t source;
  uint32_t destination;
  uint32_t count;
} GpuRawCopy;

static size_t array_bytes(size_t count, size_t elementBytes) {
  if (count == 0)
    count = 1;
  return count > SIZE_MAX / elementBytes ? 0 : count * elementBytes;
}

static size_t array_bytes_min(size_t count, size_t elementBytes,
                              size_t minimum) {
  const size_t bytes = array_bytes(count, elementBytes);
  return bytes < minimum ? minimum : bytes;
}

GeozlPivcoGpuDecoder *geozl_pivco_gpu_decoder_create(const void *module,
                                                     size_t moduleSize,
                                                     size_t elementBytes,
                                                     int device, char *error,
                                                     size_t errorCapacity) {
  if (elementBytes != 1 && elementBytes != 2) {
    geozl_gpu_error(error, errorCapacity,
                    "PivCo GPU decoding supports 1- or 2-byte elements");
    return NULL;
  }
  GeozlPivcoGpuDecoder *decoder = calloc(1, sizeof(*decoder));
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

void geozl_pivco_gpu_decoder_destroy(GeozlPivcoGpuDecoder *decoder) {
  if (decoder == NULL)
    return;
  geozl_gpu_program_destroy(decoder->program);
  free(decoder);
}

static int capture_frames(const GeozlPivcoGpuDecoder *decoder,
                          const void *const frames[], const size_t frameSizes[],
                          const size_t sizes[], size_t frameCount,
                          CapturedFrame captured[], GeozlGpuFrameShape shapes[],
                          GeozlGpuFrameShape scanShapes[], uint8_t *payload,
                          size_t payloadCapacity, uint8_t *rawValues,
                          GpuRawCopy rawCopies[], void *scratch, char *error,
                          size_t errorCapacity, size_t *payloadSize,
                          size_t *gpuCount, size_t *scanCount, size_t *rawCount,
                          size_t *rawBytes) {
  PivcoCapture capture;
  if (pivco_capture_init(&capture, payload, payloadCapacity) != 0)
    return geozl_gpu_error(error, errorCapacity,
                           "cannot create the PivCo frame parser");

  size_t outputElements = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    CapturedFrame frame = {0};
    const PivcoCaptureResult result = pivco_capture_frame(
        &capture, &frame, scratch, sizes[i], frames[i], frameSizes[i]);
    if (result == PIVCO_CAPTURE_RAW) {
      if (frame.rawElementBytes != decoder->elementBytes ||
          *rawBytes > UINT32_MAX || outputElements > UINT32_MAX ||
          sizes[i] / decoder->elementBytes > UINT32_MAX) {
        pivco_capture_destroy(&capture);
        return geozl_gpu_error(error, errorCapacity,
                               "frame %zu has a different element width", i);
      }
      memcpy(rawValues + *rawBytes, scratch, sizes[i]);
      rawCopies[(*rawCount)++] = (GpuRawCopy){
          (uint32_t)(*rawBytes / decoder->elementBytes),
          (uint32_t)outputElements,
          (uint32_t)(sizes[i] / decoder->elementBytes),
      };
      *rawBytes += sizes[i];
      outputElements += sizes[i] / decoder->elementBytes;
      continue;
    }
    if (result != PIVCO_CAPTURE_CODEC) {
      pivco_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu does not end in a supported PivCo "
                             "codec",
                             i);
    }
    captured[*gpuCount] = frame;
    const uint8_t *header = captured[*gpuCount].header;
    const int planar = frame.kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG;
    const size_t fixedHeader = planar
                                   ? PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED
                                   : TRANSPOSE_PIVCO_HEADER_FIXED;
    if (captured[*gpuCount].headerSize < fixedHeader ||
        header[8] != decoder->elementBytes) {
      pivco_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu has a different element width", i);
    }
    const uint64_t count64 = geozl_ld_le64(header);
    if (count64 == 0 || count64 > SIZE_MAX ||
        (size_t)count64 > SIZE_MAX / decoder->elementBytes ||
        (size_t)count64 * decoder->elementBytes != sizes[i]) {
      pivco_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu has inconsistent output metadata", i);
    }
    shapes[*gpuCount] = (GeozlGpuFrameShape){
        outputElements, (size_t)count64,
        planar ? geozl_ld_le32(header + 9) : (uint32_t)count64,
        planar ? geozl_ld_le32(header + 13) : 1};
    if (planar)
      scanShapes[(*scanCount)++] = shapes[*gpuCount];
    ++*gpuCount;
    outputElements += (size_t)count64;
  }
  *payloadSize = capture.payloadSize;
  pivco_capture_destroy(&capture);
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

static int build_dispatches(const PivcoGpuPlan *plan,
                            const GeozlGpuPlanePlan *planes, size_t gpuCount,
                            size_t rawCount, GeozlGpuDispatch dispatches[],
                            size_t *count, char *error, size_t errorCapacity) {
  *count = 0;
  if (plan->pivcoBlockCount != 0 &&
      append_dispatch(dispatches, count, K_DECODE_SYMBOLS,
                      plan->pivcoBlockCount, 1, error, errorCapacity))
    return 1;
  if (plan->laneBlockCount != 0 &&
      append_dispatch(dispatches, count, K_UNPACK_LANE_BLOCKS,
                      plan->laneBlockCount / 4, 1, error, errorCapacity))
    return 1;
  if (gpuCount != 0 &&
      append_dispatch(dispatches, count, K_JOIN_LANES, plan->joinCount, 1,
                      error, errorCapacity))
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

GeozlGpuMemory *geozl_pivco_gpu_decode(GeozlPivcoGpuDecoder *decoder,
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
    geozl_gpu_error(error, errorCapacity, "invalid PivCo GPU decode input");
    return NULL;
  }

  size_t *sizes = calloc(frameCount, sizeof(*sizes));
  CapturedFrame *captured = calloc(frameCount, sizeof(*captured));
  GeozlGpuFrameShape *shapes = calloc(frameCount, sizeof(*shapes));
  GeozlGpuFrameShape *scanShapes = calloc(frameCount, sizeof(*scanShapes));
  GpuRawCopy *rawCopies = calloc(frameCount, sizeof(*rawCopies));
  if (sizes == NULL || captured == NULL || shapes == NULL ||
      scanShapes == NULL || rawCopies == NULL) {
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
  if (capture_frames(decoder, frames, frameSizes, sizes, frameCount, captured,
                     shapes, scanShapes, payload, batchInfo.payloadCapacity,
                     rawValues, rawCopies, scratch, error, errorCapacity,
                     &payloadSize, &gpuCount, &scanCount, &rawCount,
                     &rawBytes)) {
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

  PivcoGpuPlan plan = {0};
  if (gpuCount != 0 && pivco_gpu_plan_build_parallel(
                           &plan, captured, shapes, gpuCount, payload,
                           payloadSize, decoder->elementBytes, 4) != 0) {
    if (plan.errorFrame == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s", plan.error);
    else if (plan.errorLane == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s in frame %zu", plan.error,
                      plan.errorFrame);
    else
      geozl_gpu_error(error, errorCapacity, "%s in lane %zu of frame %zu",
                      plan.error, plan.errorLane, plan.errorFrame);
    free(payload);
    free(rawValues);
    goto fail_arrays;
  }

  GeozlGpuPlanePlan planes = {0};
  if (scanCount != 0 &&
      geozl_gpu_plane_plan_build(&planes, scanShapes, scanCount) != 0) {
    geozl_gpu_error(error, errorCapacity, "invalid planar scan geometry");
    pivco_gpu_plan_free(&plan);
    free(payload);
    free(rawValues);
    goto fail_arrays;
  }

  const size_t laneBytes =
      plan.laneBytes <= SIZE_MAX - 256 ? plan.laneBytes + 256 : 0;
  const size_t payloadBytes = payloadSize + GEOZL_GPU_PAYLOAD_PADDING;
  const size_t resultBytes =
      batchInfo.totalBytes > laneBytes ? batchInfo.totalBytes : laneBytes;
  uint64_t dummy[4] = {0};
  GeozlGpuBuffer buffers[] = {
      {payload, payloadBytes, 1, GEOZL_GPU_COPY_IN, -1},
      {plan.symbols != NULL ? plan.symbols : (uint8_t *)dummy,
       array_bytes_min(plan.symbolCount, 1, 4), 1, GEOZL_GPU_COPY_IN, -1},
      {plan.trees != NULL ? plan.trees : (GpuTreeNode *)dummy,
       array_bytes(plan.treeCount, sizeof(GpuTreeNode)), sizeof(GpuTreeNode),
       GEOZL_GPU_COPY_IN, -1},
      {plan.nodes != NULL ? plan.nodes : (GpuNodeBits *)dummy,
       array_bytes(plan.nodeCount, sizeof(GpuNodeBits)), sizeof(GpuNodeBits),
       GEOZL_GPU_COPY_IN, -1},
      {plan.pivcoBlocks != NULL ? plan.pivcoBlocks : (GpuPivcoBlock *)dummy,
       array_bytes(plan.pivcoBlockCount, sizeof(GpuPivcoBlock)),
       sizeof(GpuPivcoBlock), GEOZL_GPU_COPY_IN, -1},
      {NULL, laneBytes, 1, 0, -1},
      {NULL, laneBytes, 1, 0, 5},
      {plan.laneBlocks != NULL ? plan.laneBlocks : (GpuLaneBlock *)dummy,
       array_bytes(plan.laneBlockCount, sizeof(GpuLaneBlock)),
       sizeof(GpuLaneBlock), GEOZL_GPU_COPY_IN, -1},
      {plan.joins != NULL ? plan.joins : (GpuJoinTask *)dummy,
       array_bytes(plan.joinCount, sizeof(GpuJoinTask)), sizeof(GpuJoinTask),
       GEOZL_GPU_COPY_IN, -1},
      {planes.tasks != NULL ? planes.tasks : (GeozlGpuPlaneTask *)dummy,
       array_bytes(planes.count, sizeof(GeozlGpuPlaneTask)),
       sizeof(GeozlGpuPlaneTask), GEOZL_GPU_COPY_IN, -1},
      {NULL, resultBytes, decoder->elementBytes, GEOZL_GPU_RESULT, -1},
      {NULL, resultBytes, 1, 0, 10},
      {NULL, resultBytes, 1, 0, 10},
      {rawCount != 0 ? rawValues : (uint8_t *)dummy,
       rawCount != 0 ? rawBytes : decoder->elementBytes, decoder->elementBytes,
       GEOZL_GPU_COPY_IN, -1},
      {rawCount != 0 ? rawCopies : (GpuRawCopy *)dummy,
       rawCount != 0 ? rawCount * sizeof(*rawCopies) : sizeof(*rawCopies),
       sizeof(*rawCopies), GEOZL_GPU_COPY_IN, -1},
  };
  if (laneBytes == 0) {
    geozl_gpu_error(error, errorCapacity, "GPU buffer size overflows");
    goto fail_plan;
  }
  for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i) {
    if (buffers[i].bytes == 0) {
      geozl_gpu_error(error, errorCapacity, "GPU buffer %zu has no storage", i);
      goto fail_plan;
    }
  }

  GeozlGpuDispatch dispatches[GEOZL_GPU_MAX_KERNELS];
  size_t dispatchCount;
  if (build_dispatches(&plan, &planes, gpuCount, rawCount, dispatches,
                       &dispatchCount, error, errorCapacity))
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
  GeozlGpuMemory *memory =
      geozl_gpu_batch_take(batch, 10, error, errorCapacity);
  geozl_gpu_batch_destroy(batch);
  if (memory != NULL && outputBytes != NULL)
    *outputBytes = batchInfo.totalBytes;

  geozl_gpu_plane_plan_free(&planes);
  pivco_gpu_plan_free(&plan);
  free(payload);
  free(rawValues);
  free(rawCopies);
  free(scanShapes);
  free(shapes);
  free(captured);
  free(sizes);
  return memory;

fail_plan:
  geozl_gpu_plane_plan_free(&planes);
  pivco_gpu_plan_free(&plan);
  free(payload);
  free(rawValues);
fail_arrays:
  free(rawCopies);
  free(scanShapes);
  free(shapes);
  free(captured);
  free(sizes);
  return NULL;
}
