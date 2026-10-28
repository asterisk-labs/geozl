#include "pivco_decode.h"

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
  GeozlGpuArena staging;
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

GeozlPivcoGpuDecoder *
geozl_pivco_gpu_decoder_create(GeozlGpuBackendId backend, const void *module,
                               size_t moduleSize, size_t elementBytes,
                               int device, char *error, size_t errorCapacity) {
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

void geozl_pivco_gpu_decoder_destroy(GeozlPivcoGpuDecoder *decoder) {
  if (decoder == NULL)
    return;
  geozl_gpu_arena_destroy(&decoder->staging);
  geozl_gpu_program_destroy(decoder->program);
  free(decoder);
}

GeozlGpuProgram *
geozl_pivco_gpu_decoder_program(const GeozlPivcoGpuDecoder *decoder) {
  return decoder == NULL ? NULL : decoder->program;
}

int pivco_gpu_frame_init(CapturedFrame *frame, GeozlGpuFrameShape *shape,
                         size_t payloadBase, size_t elementBytes,
                         size_t outputBytes, size_t outputBase) {
  const uint8_t *header = frame->header;
  const int planar = frame->kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG;
  const size_t fixedHeader = planar ? PLANAR_ZIGZAG_TRANSPOSE_PIVCO_HEADER_FIXED
                                    : TRANSPOSE_PIVCO_HEADER_FIXED;
  if (frame->headerSize < fixedHeader || header[8] != elementBytes ||
      frame->bitsOffset > SIZE_MAX - payloadBase)
    return 1;
  const uint64_t count = geozl_ld_le64(header);
  const uint32_t blockSize = geozl_ld_le32(header + (planar ? 17 : 9));
  if (count == 0 || count > SIZE_MAX / elementBytes ||
      (size_t)count * elementBytes != outputBytes ||
      (blockSize < GEOZL_GPU_PIVCO_MIN_BLOCK && blockSize < count))
    return 1;
  frame->bitsOffset += payloadBase;
  *shape =
      (GeozlGpuFrameShape){outputBase, (size_t)count,
                           planar ? geozl_ld_le32(header + 9) : (uint32_t)count,
                           planar ? geozl_ld_le32(header + 13) : 1};
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

static int build_dispatches(PivcoGpuWork *work, size_t gpuCount,
                            size_t rawCount, char *error,
                            size_t errorCapacity) {
  const PivcoGpuPlan *plan = &work->plan;
  const GeozlGpuPlanePlan *planes = &work->planes;
  GeozlGpuDispatch *dispatches = work->dispatches;
  size_t *count = &work->dispatchCount;
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

int pivco_gpu_work_build(PivcoGpuWork *work, GeozlGpuArena *staging,
                         size_t elementBytes, const CapturedFrame frames[],
                         const GeozlGpuFrameShape shapes[], size_t frameCount,
                         const size_t frameIds[], uint8_t *payload,
                         size_t payloadSize, const uint8_t *rawValues,
                         size_t rawBytes, const GeozlGpuRawCopy rawCopies[],
                         size_t rawCount, size_t outputBytes, size_t threads,
                         char *error, size_t errorCapacity) {
  *work = (PivcoGpuWork){0};
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

  if (frameCount != 0 && pivco_gpu_plan_build_parallel(
                             &work->plan, frames, shapes, frameCount, payload,
                             payloadSize, elementBytes, threads, &alloc) != 0) {
    const PivcoGpuPlan *plan = &work->plan;
    if (plan->errorFrame == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s", plan->error);
    else if (plan->errorLane == SIZE_MAX)
      geozl_gpu_error(error, errorCapacity, "%s in frame %zu", plan->error,
                      frame_id(frameIds, plan->errorFrame));
    else
      geozl_gpu_error(error, errorCapacity, "%s in lane %zu of frame %zu",
                      plan->error, plan->errorLane,
                      frame_id(frameIds, plan->errorFrame));
    return 1;
  }

  GeozlGpuFrameShape *scans =
      frameCount == 0 ? NULL : malloc(frameCount * sizeof(*scans));
  if (frameCount != 0 && scans == NULL) {
    pivco_gpu_work_free(work);
    return geozl_gpu_error(error, errorCapacity, "out of memory");
  }
  size_t scanCount = 0;
  for (size_t i = 0; i < frameCount; ++i) {
    if (frames[i].kind == PIVCO_CAPTURE_KIND_PLANAR_ZIGZAG)
      scans[scanCount++] = shapes[i];
  }
  const int planned =
      scanCount == 0 ||
      geozl_gpu_plane_plan_build(&work->planes, scans, scanCount, &alloc) == 0;
  free(scans);
  if (!planned) {
    pivco_gpu_work_free(work);
    return geozl_gpu_error(error, errorCapacity,
                           "invalid planar scan geometry");
  }

  const PivcoGpuPlan *plan = &work->plan;
  const GeozlGpuPlanePlan *planes = &work->planes;
  const size_t laneBytes = plan->laneBytes + 256;
  work->resultBytes = outputBytes > laneBytes ? outputBytes : laneBytes;
  GeozlGpuBuffer *buffers = work->buffers;
  buffers[0] = geozl_gpu_arena_input(
      staging, payload, payloadSize + GEOZL_GPU_PAYLOAD_PADDING, 1);
  buffers[1] = geozl_gpu_arena_input(
      staging, plan->symbols != NULL ? (void *)plan->symbols : dummy,
      plan->symbolCount < 4 ? 4 : plan->symbolCount, 1);
  buffers[2] = geozl_gpu_arena_input(
      staging, plan->trees != NULL ? (void *)plan->trees : dummy,
      plan->treeCount, sizeof(GpuTreeNode));
  buffers[3] = geozl_gpu_arena_input(
      staging, plan->nodes != NULL ? (void *)plan->nodes : dummy,
      plan->nodeCount, sizeof(GpuNodeBits));
  buffers[4] = geozl_gpu_arena_input(
      staging, plan->pivcoBlocks != NULL ? (void *)plan->pivcoBlocks : dummy,
      plan->pivcoBlockCount, sizeof(GpuPivcoBlock));
  buffers[5] =
      (GeozlGpuBuffer){.bytes = laneBytes, .elementBytes = 1, .alias = -1};
  buffers[6] =
      (GeozlGpuBuffer){.bytes = laneBytes, .elementBytes = 1, .alias = 5};
  buffers[7] = geozl_gpu_arena_input(
      staging, plan->laneBlocks != NULL ? (void *)plan->laneBlocks : dummy,
      plan->laneBlockCount, sizeof(GpuLaneBlock));
  buffers[8] = geozl_gpu_arena_input(
      staging, plan->joins != NULL ? (void *)plan->joins : dummy,
      plan->joinCount, sizeof(GpuJoinTask));
  buffers[9] = geozl_gpu_arena_input(
      staging, planes->tasks != NULL ? (void *)planes->tasks : dummy,
      planes->count, sizeof(GeozlGpuPlaneTask));
  buffers[PIVCO_GPU_OUTPUT] = (GeozlGpuBuffer){.bytes = work->resultBytes,
                                               .elementBytes = elementBytes,
                                               .flags = GEOZL_GPU_RESULT,
                                               .alias = -1};
  buffers[11] = (GeozlGpuBuffer){
      .bytes = work->resultBytes, .elementBytes = 1, .alias = PIVCO_GPU_OUTPUT};
  buffers[12] = buffers[11];
  buffers[13] = geozl_gpu_arena_input(
      staging, rawValues != NULL ? (const void *)rawValues : dummy,
      rawBytes / elementBytes, elementBytes);
  buffers[14] = geozl_gpu_arena_input(
      staging, rawCount != 0 ? (const void *)rawCopies : dummy, rawCount,
      sizeof(*rawCopies));
  if (build_dispatches(work, frameCount, rawCount, error, errorCapacity)) {
    pivco_gpu_work_free(work);
    return 1;
  }
  return 0;
}

void pivco_gpu_work_free(PivcoGpuWork *work) {
  geozl_gpu_plane_plan_free(&work->planes);
  pivco_gpu_plan_free(&work->plan);
}

// Raw frames are rare, so their staging is taken at the first one, sized for
// the output still to come.
static int capture_frames(GeozlPivcoGpuDecoder *decoder,
                          const void *const frames[], const size_t frameSizes[],
                          const size_t sizes[], size_t frameCount,
                          size_t totalBytes, CapturedFrame captured[],
                          GeozlGpuFrameShape shapes[], uint8_t *payload,
                          size_t payloadCapacity, uint8_t **rawValues,
                          GeozlGpuRawCopy rawCopies[], void *scratch,
                          char *error, size_t errorCapacity,
                          size_t *payloadSize, size_t *gpuCount,
                          size_t *rawCount, size_t *rawBytes) {
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
      if (*rawValues == NULL &&
          (*rawValues = geozl_gpu_arena_alloc(
               &decoder->staging,
               totalBytes - outputElements * decoder->elementBytes)) == NULL) {
        pivco_capture_destroy(&capture);
        return geozl_gpu_error(error, errorCapacity, "%s",
                               geozl_gpu_arena_error(&decoder->staging));
      }
      memcpy(*rawValues + *rawBytes, scratch, sizes[i]);
      rawCopies[(*rawCount)++] = (GeozlGpuRawCopy){
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
    if (pivco_gpu_frame_init(&captured[*gpuCount], &shapes[*gpuCount], 0,
                             decoder->elementBytes, sizes[i],
                             outputElements) != 0) {
      pivco_capture_destroy(&capture);
      return geozl_gpu_error(error, errorCapacity,
                             "frame %zu has inconsistent output metadata", i);
    }
    ++*gpuCount;
    outputElements += sizes[i] / decoder->elementBytes;
  }
  *payloadSize = capture.payloadSize;
  pivco_capture_destroy(&capture);
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

  // Kernel inputs are built in staging, which the device reads in place.
  GeozlGpuArena *staging = &decoder->staging;
  geozl_gpu_arena_reset(staging);
  const GeozlGpuHostAlloc alloc = geozl_gpu_arena_allocator(staging);
  PivcoGpuWork work = {0};
  GeozlGpuMemory *memory = NULL;
  void *scratch = NULL;
  size_t *sizes = calloc(frameCount, sizeof(*sizes));
  CapturedFrame *captured = calloc(frameCount, sizeof(*captured));
  GeozlGpuFrameShape *shapes = calloc(frameCount, sizeof(*shapes));
  if (sizes == NULL || captured == NULL || shapes == NULL) {
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
                     batchInfo.totalBytes, captured, shapes, payload,
                     batchInfo.payloadCapacity, &rawValues, rawCopies, scratch,
                     error, errorCapacity, &payloadSize, &gpuCount, &rawCount,
                     &rawBytes) ||
      pivco_gpu_work_build(&work, staging, decoder->elementBytes, captured,
                           shapes, gpuCount, NULL, payload, payloadSize,
                           rawValues, rawBytes, rawCopies, rawCount,
                           batchInfo.totalBytes, 4, error, errorCapacity))
    goto done;

  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      decoder->program, work.buffers, PIVCO_GPU_BUFFERS, error, errorCapacity);
  if (batch == NULL)
    goto done;
  GeozlGpuRunTimes times;
  if (geozl_gpu_batch_run(batch, work.dispatches, work.dispatchCount, &times,
                          error, errorCapacity) == 0)
    memory =
        geozl_gpu_batch_take(batch, PIVCO_GPU_OUTPUT, error, errorCapacity);
  geozl_gpu_batch_destroy(batch);
  if (memory != NULL && outputBytes != NULL)
    *outputBytes = batchInfo.totalBytes;

done:
  pivco_gpu_work_free(&work);
  free(scratch);
  free(shapes);
  free(captured);
  free(sizes);
  return memory;
}
