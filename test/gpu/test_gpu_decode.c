// Decodes batches that mix device and host codecs through geozl/gpu.h and
// checks every copied sample against the CPU decoder.

#include "geozl/dtype.h"
#include "geozl/geozl.h"
#include "geozl/gpu.h"

#include "capture_frame_builder.h"

#include "common/endian.h"
#include "gpu/runtime.h"
#include "pfor/graph_pfor.h"
#include "pivco_lanes/pivco_lanes.h"
#include "planar_zigzag_pfor/graph_planar_zigzag_pfor.h"
#include "planar_zigzag_transpose_pivco/graph_planar_zigzag_transpose_pivco.h"
#include "transpose_pivco/graph_transpose_pivco.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_input.h"
#include "openzl/zl_segmenter.h"

#ifndef __APPLE__
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#define DEVICE GEOZL_GPU_DEVICE_METAL
#else
#define DEVICE GEOZL_GPU_DEVICE_CUDA
#endif

#define MAX_FRAMES 64
#define MAX_COPIES 256

typedef struct {
  uint32_t planes, rows, width;
} Shape;

typedef struct {
  uint8_t *data[MAX_FRAMES];
  GeozlGpuFrame frames[MAX_FRAMES];
  uint8_t *decoded[MAX_FRAMES];
  size_t count;
  GeozlGpuCopy copies[MAX_COPIES];
  size_t copyCount;
  size_t targetBytes;
} Batch;

static const Shape shapes[] = {{3, 20, 300}, {1, 64, 64}, {2, 7, 513}};

static size_t round_up_test(size_t bytes) { return (bytes + 63) / 64 * 64; }

static uint64_t state = 0x9e3779b97f4a7c15u;

static uint32_t next_random(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (uint32_t)state;
}

// Smooth rows with a little noise, the kind of data the predictors expect.
static void fill(uint8_t *out, size_t count, size_t width, size_t bytes) {
  uint32_t value = 1000;
  for (size_t i = 0; i < count; ++i) {
    if (i % width == 0)
      value = 1000 + next_random() % 64;
    value += next_random() % 7 - 3;
    if (bytes == 1)
      out[i] = (uint8_t)value;
    else if (bytes == 2)
      ((uint16_t *)out)[i] = (uint16_t)value;
    else
      ((uint32_t *)out)[i] = value;
  }
}

static int dtype_of(size_t bytes) {
  return bytes == 1 ? GEOZL_DT_U8 : bytes == 2 ? GEOZL_DT_U16 : GEOZL_DT_U32;
}

static int add_frame(Batch *batch, const char *recipe, const Shape *shape,
                     size_t bytes) {
  const size_t count = (size_t)shape->planes * shape->rows * shape->width;
  uint8_t *source = malloc(count * bytes);
  const size_t bound = ZL_compressBound(count * bytes);
  uint8_t *frame = malloc(bound);
  char error[256];
  geozl_2d_graph *graph = NULL;
  size_t size = 0;
  if (source == NULL || frame == NULL) {
    free(source);
    free(frame);
    return 1;
  }
  fill(source, count, shape->width, bytes);
  if (geozl_2d_graph_open_c(&graph, recipe, shape->width, shape->planes, NULL,
                            dtype_of(bytes), GEOZL_NODATA_NONE, 0, source,
                            count, bytes, error, sizeof(error)) != 0 ||
      geozl_2d_compress_graph_c(graph, source, count, frame, bound, &size,
                                error, sizeof(error)) != 0) {
    fprintf(stderr, "%s: %s\n", recipe, error);
    geozl_2d_graph_close_c(graph);
    free(source);
    free(frame);
    return 1;
  }
  geozl_2d_graph_close_c(graph);
  const size_t i = batch->count++;
  batch->data[i] = frame;
  batch->decoded[i] = source;
  batch->frames[i] = (GeozlGpuFrame){frame, size, count * bytes};
  return 0;
}

// Each frame lands whole, then cropped from its last plane into padded rows,
// then every other sample of its first plane.
static void add_copies(Batch *batch, size_t frame, const Shape *shape,
                       size_t bytes) {
  const size_t row = shape->width * bytes;
  const size_t plane = shape->rows * row;
  GeozlGpuCopy *copy = &batch->copies[batch->copyCount];
  copy[0] = (GeozlGpuCopy){frame,
                           0,
                           row,
                           bytes,
                           batch->targetBytes,
                           row,
                           bytes,
                           shape->width,
                           shape->planes * shape->rows};
  batch->targetBytes += plane * shape->planes;
  const uint32_t width = shape->width - 5, rows = shape->rows - 3;
  copy[1] = (GeozlGpuCopy){frame,
                           (shape->planes - 1) * plane + 2 * row + 3 * bytes,
                           row,
                           bytes,
                           batch->targetBytes,
                           (width + 5) * bytes,
                           bytes,
                           width,
                           rows};
  batch->targetBytes += (size_t)(width + 5) * rows * bytes;
  copy[2] = (GeozlGpuCopy){
      frame,   row,       row,          bytes,          batch->targetBytes,
      2 * row, 2 * bytes, shape->width, shape->rows - 1};
  batch->targetBytes += 2 * plane;
  batch->copyCount += 3;
}

static void add_whole_copy(Batch *batch, size_t frame, size_t bytes) {
  const size_t size = batch->frames[frame].decodedBytes;
  batch->copies[batch->copyCount++] = (GeozlGpuCopy){frame,
                                                     0,
                                                     size,
                                                     bytes,
                                                     batch->targetBytes,
                                                     size,
                                                     bytes,
                                                     (uint32_t)(size / bytes),
                                                     1};
  batch->targetBytes += size;
}

static void expect(const Batch *batch, size_t sampleBytes, uint8_t *want,
                   uint8_t *written) {
  for (size_t c = 0; c < batch->copyCount; ++c) {
    const GeozlGpuCopy *copy = &batch->copies[c];
    for (uint32_t r = 0; r < copy->rows; ++r) {
      for (uint32_t x = 0; x < copy->width; ++x) {
        const size_t from =
            copy->source + r * copy->sourceRow + x * copy->sourceStep;
        const size_t to = copy->destination + r * copy->destinationRow +
                          x * copy->destinationStep;
        memcpy(want + to, batch->decoded[copy->frame] + from, sampleBytes);
        memset(written + to, 1, sampleBytes);
      }
    }
  }
}

// The second pass writes behind an offset into the same memory.
static int run_shifted(GeozlGpu *gpu, const Batch *batch, size_t sampleBytes,
                       size_t elementBytes, const char *name,
                       const uint8_t forceHost[], size_t shift) {
  char error[512];
  GeozlGpuMemory *memory =
      geozl_gpu_alloc(gpu, batch->targetBytes + shift, error, sizeof(error));
  uint8_t *got = malloc(batch->targetBytes + shift);
  uint8_t *want = calloc(1, batch->targetBytes);
  uint8_t *written = calloc(1, batch->targetBytes);
  int failed = 1;
  if (memory == NULL || got == NULL || want == NULL || written == NULL) {
    fprintf(stderr, "%s: allocation failed: %s\n", name, error);
    goto done;
  }
  expect(batch, sampleBytes, want, written);
  for (int pass = 0; pass < 2; ++pass) {
    const size_t offset = pass == 0 ? 0 : shift;
    const GeozlGpuTarget target = {geozl_gpu_pointer(memory), offset,
                                   batch->targetBytes, NULL};
    if (geozl_gpu_decode_ex(gpu, batch->frames, batch->count, batch->copies,
                            batch->copyCount, sampleBytes, elementBytes,
                            &target, forceHost, error, sizeof(error)) != 0 ||
        geozl_gpu_synchronize(gpu, error, sizeof(error)) != 0 ||
        geozl_gpu_memory_download(memory, got, batch->targetBytes + shift,
                                  error, sizeof(error)) != 0) {
      fprintf(stderr, "%s: %s\n", name, error);
      goto done;
    }
    for (size_t i = 0; i < batch->targetBytes; ++i) {
      if (written[i] && got[offset + i] != want[i]) {
        fprintf(stderr, "%s: byte %zu is %u, want %u\n", name, i,
                got[offset + i], want[i]);
        goto done;
      }
    }
  }
  failed = 0;

done:
  geozl_gpu_free(memory);
  free(written);
  free(want);
  free(got);
  return failed;
}

static int run_with_host_flags(GeozlGpu *gpu, const Batch *batch,
                               size_t sampleBytes, size_t elementBytes,
                               const char *name, const uint8_t forceHost[]) {
  return run_shifted(gpu, batch, sampleBytes, elementBytes, name, forceHost,
                     64 * sampleBytes);
}

static int run(GeozlGpu *gpu, const Batch *batch, size_t sampleBytes,
               size_t elementBytes, const char *name) {
  return run_with_host_flags(gpu, batch, sampleBytes, elementBytes, name, NULL);
}

static void clear(Batch *batch) {
  for (size_t i = 0; i < batch->count; ++i) {
    free(batch->data[i]);
    free(batch->decoded[i]);
  }
  memset(batch, 0, sizeof(*batch));
}

static int check_mixed(GeozlGpu *gpu, size_t bytes) {
  static const char *const recipes[] = {
      "planar>zigzag>pfor",
      "id>pfor",
      "planar>zigzag>transpose>pivco",
      "id>transpose>pivco",
      "planar>zigzag>zstd",
      "delta_1d>pfor",
  };
  Batch batch = {0};
  int failed = 0;
  for (size_t r = 0; r < sizeof(recipes) / sizeof(recipes[0]) && !failed; ++r) {
    if (bytes == 1 && strstr(recipes[r], "transpose") != NULL &&
        strstr(recipes[r], "planar") == NULL)
      continue;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
      if (add_frame(&batch, recipes[r], &shapes[s], bytes) != 0) {
        failed = 1;
        break;
      }
      add_copies(&batch, batch.count - 1, &shapes[s], bytes);
    }
  }
  char name[32];
  snprintf(name, sizeof(name), "mixed u%zu", bytes * 8);
  if (!failed)
    failed = run(gpu, &batch, bytes, bytes, name);
  clear(&batch);
  return failed;
}

// Four-byte samples and frames read as pairs of two-byte samples stay on the
// host path but still go through the place kernel.
static int check_host_widths(GeozlGpu *gpu) {
  Batch batch = {0};
  const Shape shape = {2, 9, 40};
  int failed = add_frame(&batch, "planar>zigzag>zstd", &shape, 4) ||
               add_frame(&batch, "planar>zigzag>pfor", &shape, 4);
  for (size_t i = 0; !failed && i < batch.count; ++i)
    add_copies(&batch, i, &shape, 4);
  if (!failed)
    failed = run(gpu, &batch, 4, 4, "u32");
  clear(&batch);

  if (!failed)
    failed = add_frame(&batch, "planar>zigzag>pfor", &shape, 2);
  if (!failed) {
    const Shape pairs = {2, 9, 20};
    add_copies(&batch, 0, &pairs, 4);
    failed = run(gpu, &batch, 4, 2, "u16 pairs");
  }
  clear(&batch);
  return failed;
}

static int refuses(GeozlGpu *gpu, const Batch *batch, const GeozlGpuCopy *copy,
                   size_t decodedBytes, const char *name) {
  char error[512];
  GeozlGpuMemory *memory = geozl_gpu_alloc(gpu, 4096, error, sizeof(error));
  if (memory == NULL) {
    fprintf(stderr, "%s: %s\n", name, error);
    return 1;
  }
  GeozlGpuFrame frame = batch->frames[0];
  frame.decodedBytes = decodedBytes;
  const GeozlGpuTarget target = {geozl_gpu_pointer(memory), 0, 4096, NULL};
  const int decoded = geozl_gpu_decode(gpu, &frame, 1, copy, 1, 2, 2, &target,
                                       error, sizeof(error)) == 0;
  geozl_gpu_synchronize(gpu, NULL, 0);
  geozl_gpu_free(memory);
  if (decoded || error[0] == '\0') {
    fprintf(stderr, "%s was accepted\n", name);
    return 1;
  }
  return 0;
}

static int check_refusals(GeozlGpu *gpu) {
  Batch batch = {0};
  const Shape shape = {1, 16, 32};
  if (add_frame(&batch, "planar>zigzag>pfor", &shape, 2) != 0)
    return 1;
  const size_t bytes = batch.frames[0].decodedBytes;
  GeozlGpuCopy copy = {0, 0, 64, 2, 0, 64, 2, 32, 16};
  int failed = refuses(gpu, &batch, &copy, bytes - 2, "a short frame size");
  copy.rows = 17;
  failed |= refuses(gpu, &batch, &copy, bytes, "a copy past its frame");
  copy.rows = 16;
  copy.destination = 4096 - 64;
  failed |= refuses(gpu, &batch, &copy, bytes, "a copy past the target");
  copy.destination = 1;
  failed |= refuses(gpu, &batch, &copy, bytes, "an unaligned copy");
  copy.destination = 0;
  GeozlGpuCopy tall = {0, 0, 0, 2, 0, 0, 2, 1, UINT32_MAX};
  failed |= refuses(gpu, &batch, &tall, bytes, "a copy repeating one row");
  GeozlGpuCopy wide = {0, 0, 0, 0, 0, 0, 0, UINT32_MAX, 1};
  failed |= refuses(gpu, &batch, &wide, bytes, "a copy repeating one sample");
  batch.data[0][batch.frames[0].size / 2] ^= 0x5a;
  failed |= refuses(gpu, &batch, &copy, bytes, "a corrupted frame");
  clear(&batch);
  return failed;
}

static int check_capture_errors(GeozlGpu *gpu) {
  uint8_t frame[1024];
  uint8_t header[TRANSPOSE_PIVCO_HEADER_FIXED + GEOZL_PIVCO_LANE_HEADER_SIZE] =
      {0};
  geozl_st_le64(header, 1024);
  header[8] = 1;
  geozl_st_le32(header + 9, 32768);
  header[13] = GEOZL_PIVCO_LANE_PIVCO;
  geozl_st_le16(header + 14, 1);

  const CaptureFramePlan plans[] = {
      {.ctid = GEOZL_CTID_TRANSPOSE_PIVCO,
       .header = header,
       .headerSize = sizeof(header),
       .outputCount = 2,
       .numericOutputs = 1,
       .outputBytes = {1, 0}},
      {.ctid = GEOZL_CTID_PFOR,
       .header = header,
       .headerSize = PFOR_HEADER_SIZE,
       .outputCount = 1,
       .outputBytes = {1, 0}},
  };
  char error[512];
  GeozlGpuMemory *memory = geozl_gpu_alloc(gpu, 64, error, sizeof(error));
  if (memory == NULL)
    return 1;
  const GeozlGpuTarget target = {geozl_gpu_pointer(memory), 0, 64, NULL};
  const GeozlGpuCopy copy = {0, 0, 0, 1, 0, 0, 1, 64, 1};
  int failed = 0;
  for (size_t i = 0; i < sizeof(plans) / sizeof(plans[0]); ++i) {
    const size_t frameSize =
        capture_test_build_frame(frame, sizeof(frame), 64, &plans[i]);
    const GeozlGpuFrame input = {frame, frameSize, 64};
    error[0] = '\0';
    if (frameSize == 0 ||
        geozl_gpu_decode(gpu, &input, 1, &copy, 1, 1, 1, &target, error,
                         sizeof(error)) == 0 ||
        strstr(error, "invalid compressed frame") == NULL) {
      fprintf(stderr, "an inconsistent GPU codec header was accepted: %s\n",
              error);
      failed = 1;
      break;
    }
  }
  geozl_gpu_free(memory);
  return failed;
}

enum { SEGMENT_SAMPLES = 32768 };
static ZL_GraphID segmentGraphs[2];

static ZL_Report split_frame(ZL_Segmenter *ctx) {
  for (size_t i = 0; i < 2; ++i) {
    const ZL_Input *input = ZL_Segmenter_getInput(ctx, 0);
    if (ZL_Input_numElts(input) < SEGMENT_SAMPLES)
      return ZL_returnError(ZL_ErrorCode_GENERIC);
    size_t count = SEGMENT_SAMPLES;
    ZL_Report report =
        ZL_Segmenter_processChunk(ctx, &count, 1, segmentGraphs[i], NULL);
    if (ZL_isError(report))
      return report;
  }
  return ZL_returnSuccess();
}

static int add_segmented_frame(Batch *batch, size_t bytes, int firstStore,
                               int secondStore) {
  enum { COUNT = 2 * SEGMENT_SAMPLES };
  const size_t outputBytes = COUNT * bytes;
  const size_t bound = ZL_compressBound(outputBytes);
  uint8_t *source = malloc(outputBytes);
  uint8_t *frame = malloc(bound);
  ZL_Compressor *compressor = ZL_Compressor_create();
  ZL_CCtx *cctx = ZL_CCtx_create();
  ZL_TypedRef *ref = NULL;
  int failed = 1;
  if (source == NULL || frame == NULL || compressor == NULL || cctx == NULL)
    goto done;
  fill(source, COUNT, 256, bytes);
  ref = ZL_TypedRef_createNumeric(source, bytes, COUNT);
  if (ref == NULL)
    goto done;
  const ZL_NodeID node = bytes == 1 ? geozl_node_pfor(compressor)
                                    : geozl_node_transpose_pivco(compressor);
  const ZL_GraphID codec =
      bytes == 1 ? ZL_Compressor_registerStaticGraph_fromNode(
                       compressor, node, ZL_GRAPHLIST(ZL_GRAPH_STORE))
                 : ZL_Compressor_registerStaticGraph_fromNode(
                       compressor, node,
                       ZL_GRAPHLIST(ZL_GRAPH_STORE, ZL_GRAPH_STORE));
  segmentGraphs[0] = firstStore ? ZL_GRAPH_STORE : codec;
  segmentGraphs[1] = secondStore ? ZL_GRAPH_STORE : codec;
  const ZL_SegmenterDesc desc = {
      .name = "two chunks",
      .segmenterFn = split_frame,
      .inputTypeMasks = (const ZL_Type[]){ZL_Type_numeric},
      .numInputs = 1,
      .customGraphs = segmentGraphs,
      .numCustomGraphs = 2};
  const ZL_GraphID graph = ZL_Compressor_registerSegmenter(compressor, &desc);
  if (ZL_isError(ZL_Compressor_selectStartingGraphID(compressor, graph)) ||
      ZL_isError(ZL_CCtx_refCompressor(cctx, compressor)) ||
      ZL_isError(ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion,
                                     ZL_MAX_FORMAT_VERSION)))
    goto done;
  const ZL_Report report = ZL_CCtx_compressTypedRef(cctx, frame, bound, ref);
  if (ZL_isError(report)) {
    fprintf(stderr, "segmented compression: %s\n",
            ZL_CCtx_getErrorContextString(cctx, report));
    goto done;
  }
  const size_t i = batch->count++;
  batch->data[i] = frame;
  batch->decoded[i] = source;
  batch->frames[i] = (GeozlGpuFrame){frame, ZL_validResult(report), outputBytes};
  frame = NULL;
  source = NULL;
  failed = 0;
done:
  ZL_TypedRef_free(ref);
  ZL_CCtx_free(cctx);
  ZL_Compressor_free(compressor);
  free(frame);
  free(source);
  return failed;
}

static int check_segmented_frames(GeozlGpu *gpu, size_t bytes) {
  static const uint8_t stores[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  Batch batch = {0};
  int failed = 0;
  // First and second chunks independently use the GPU codec or STORE.
  for (size_t i = 0; i < sizeof(stores) / sizeof(stores[0]) && !failed; ++i) {
    failed = add_segmented_frame(&batch, bytes, stores[i][0], stores[i][1]);
    if (!failed)
      add_whole_copy(&batch, batch.count - 1, bytes);
  }
  if (!failed)
    failed = run(gpu, &batch, bytes, bytes, "segmented");
  clear(&batch);
  return failed;
}

static int check_forced_host(GeozlGpu *gpu) {
  Batch batch = {0};
  const Shape shape = {1, 64, 64};
  int failed = add_frame(&batch, "id>pfor", &shape, 1) ||
               add_frame(&batch, "planar>zigzag>pfor", &shape, 1);
  for (size_t i = 0; i < batch.count; ++i)
    add_whole_copy(&batch, i, 1);
  const uint8_t forceHost[] = {1, 0};
  if (!failed)
    failed = run_with_host_flags(gpu, &batch, 1, 1, "forced host", forceHost);
  clear(&batch);
  return failed;
}

// Frames past a backend limit of the scan kernels still decode. The second one
// needs more grid groups than CUDA allows.
static int check_plane_limits(void) {
  char error[512];
  GeozlGpu *gpu =
      geozl_gpu_create(DEVICE, 0, 2, 16u << 20, 1, error, sizeof(error));
  if (gpu == NULL) {
    fprintf(stderr, "plane limits: %s\n", error);
    return 1;
  }
  const Shape limits[] = {{65536, 1, 1}, {1, 1, 8388481}};
  Batch batch = {0};
  int failed = 0;
  for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]) && !failed; ++i) {
    failed = add_frame(&batch, "planar>zigzag>pfor", &limits[i], 1);
    if (!failed)
      add_whole_copy(&batch, i, 1);
  }
  if (!failed)
    failed = run(gpu, &batch, 1, 1, "plane limits");
  clear(&batch);
  geozl_gpu_destroy(gpu);
  return failed;
}

static int check_batch_limit(GeozlGpu *gpu) {
  Batch batch = {0};
  const Shape shape = {1, 16, 32};
  if (add_frame(&batch, "planar>zigzag>pfor", &shape, 2) != 0)
    return 1;
  GeozlGpuFrame frames[2] = {batch.frames[0], batch.frames[0]};
  frames[0].decodedBytes = 600 * 1024;
  frames[1].decodedBytes = 600 * 1024;
  const GeozlGpuCopy copy = {0, 0, 0, 2, 0, 0, 2, 1, 1};
  char error[512];
  GeozlGpuMemory *memory = geozl_gpu_alloc(gpu, 2, error, sizeof(error));
  if (memory == NULL) {
    clear(&batch);
    return 1;
  }
  const GeozlGpuTarget target = {geozl_gpu_pointer(memory), 0, 2, NULL};
  const int failed = geozl_gpu_decode(gpu, frames, 2, &copy, 1, 2, 2, &target,
                                      error, sizeof(error)) == 0 ||
                     strstr(error, "batch limit") == NULL;
  if (failed)
    fprintf(stderr, "an oversized GPU batch was accepted: %s\n", error);
  geozl_gpu_free(memory);
  clear(&batch);
  return failed;
}

// Check direct output, PivCo scratch, misaligned targets and overlapping copies.
static int check_direct(GeozlGpu *gpu, size_t bytes) {
  static const char *const recipes[] = {
      "planar>zigzag>pfor", "id>pfor", "planar>zigzag>transpose>pivco"};
  const Shape shape = {2, 64, 64};
  for (size_t mix = 0; mix < 3; ++mix) {
    Batch batch = {0};
    int failed = 0;
    for (size_t r = 0; r < 3 && !failed; ++r) {
      // First two cases use one codec; the last mixes PFOR and PivCo.
      if (mix != 2 && (mix == 0) != (r == 2))
        continue;
      for (size_t n = 0; n < 2 && !failed; ++n) {
        failed = add_frame(&batch, recipes[r], &shape, bytes);
        if (!failed)
          add_whole_copy(&batch, batch.count - 1, bytes);
      }
    }
    if (!failed)
      failed = run(gpu, &batch, bytes, bytes, "direct");
    if (!failed)
      failed = run_shifted(gpu, &batch, bytes, bytes, "off alignment", NULL,
                           2 * bytes);
    if (!failed && mix == 2) {
      // Keep the copy contiguous while splitting it into rows.
      for (size_t c = 0; c < batch.copyCount; ++c) {
        GeozlGpuCopy *copy = &batch.copies[c];
        const size_t row = shape.width * bytes;
        copy->sourceRow = row;
        copy->destinationRow = row;
        copy->width = shape.width;
        copy->rows = shape.planes * shape.rows;
      }
      failed = run(gpu, &batch, bytes, bytes, "direct rows");
    }
    clear(&batch);
    if (failed)
      return 1;
  }

  // Mix a host fallback frame with direct frames.
  Batch back = {0};
  int failed = add_frame(&back, recipes[0], &shape, bytes) ||
               add_frame(&back, "delta_1d>pfor", &shape, bytes) ||
               add_frame(&back, recipes[2], &shape, bytes);
  for (size_t i = 0; i < back.count && !failed; ++i)
    add_whole_copy(&back, i, bytes);
  if (!failed)
    failed = run(gpu, &back, bytes, bytes, "direct with a host frame");
  clear(&back);
  if (failed)
    return 1;

  Batch same = {0};
  failed = add_frame(&same, recipes[0], &shape, bytes);
  if (!failed) {
    add_whole_copy(&same, 0, bytes);
    same.frames[1] = same.frames[0];
    same.decoded[1] = same.decoded[0];
    same.count = 2;
    same.copies[1] = same.copies[0];
    same.copies[1].frame = 1;
    same.copyCount = 2;
    failed = run(gpu, &same, bytes, bytes, "overlapping frames");
    same.count = 1;
    same.decoded[1] = NULL;
  }
  clear(&same);
  return failed;
}

#ifdef __APPLE__
typedef struct {
  void *stream[2];
} Streams;
static void streams_open(Streams *streams) {
  *streams = (Streams){{NULL, NULL}};
}
static void streams_close(Streams *streams) { (void)streams; }
#else
typedef struct {
  void *stream[2];
  void *library;
  int (*destroy)(void *);
} Streams;

// Two streams of device 0, for calls that move between them. Without them the
// checks below run on the default stream alone.
static void streams_open(Streams *streams) {
  *streams = (Streams){0};
  streams->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (streams->library == NULL)
    return;
  int (*init)(unsigned) = (int (*)(unsigned))dlsym(streams->library, "cuInit");
  int (*device)(int *, int) =
      (int (*)(int *, int))dlsym(streams->library, "cuDeviceGet");
  int (*retain)(void **, int) = (int (*)(void **, int))dlsym(
      streams->library, "cuDevicePrimaryCtxRetain");
  int (*current)(void *) =
      (int (*)(void *))dlsym(streams->library, "cuCtxSetCurrent");
  int (*create)(void **, unsigned) =
      (int (*)(void **, unsigned))dlsym(streams->library, "cuStreamCreate");
  streams->destroy =
      (int (*)(void *))dlsym(streams->library, "cuStreamDestroy_v2");
  int id = 0;
  void *context = NULL;
  if (init == NULL || device == NULL || retain == NULL || current == NULL ||
      create == NULL || streams->destroy == NULL || init(0) != 0 ||
      device(&id, 0) != 0 || retain(&context, id) != 0 ||
      current(context) != 0)
    return;
  for (size_t i = 0; i < 2; ++i) {
    if (create(&streams->stream[i], 0) != 0)
      streams->stream[i] = NULL;
  }
}

static void streams_close(Streams *streams) {
  for (size_t i = 0; i < 2; ++i) {
    if (streams->stream[i] != NULL)
      streams->destroy(streams->stream[i]);
  }
  if (streams->library != NULL)
    dlclose(streams->library);
}
#endif

// Check overlapping calls, scratch growth, stream changes and failure cleanup.
static int check_waves(GeozlGpu *gpu, size_t bytes) {
  enum { WAVES = 6 };
  static const Shape growing[WAVES] = {{1, 16, 32},  {2, 7, 513},
                                       {1, 64, 64},  {3, 20, 300},
                                       {1, 256, 512}, {2, 64, 513}};
  static const char *const recipes[] = {"planar>zigzag>pfor",
                                        "planar>zigzag>transpose>pivco"};
  Batch batches[WAVES];
  memset(batches, 0, sizeof(batches));
  size_t offsets[WAVES + 1] = {0};
  char error[512];
  GeozlGpuMemory *memories[4] = {NULL};
  Streams streams;
  streams_open(&streams);
  uint8_t *got = NULL, *want = NULL, *written = NULL;
  int failed = 0;
#ifndef __APPLE__
  if (streams.stream[0] == NULL || streams.stream[1] == NULL) {
    fprintf(stderr, "cannot create the CUDA streams\n");
    failed = 1;
  }
#endif
  for (size_t i = 0; i < WAVES && !failed; ++i) {
    for (size_t r = 0; r < 2 && !failed; ++r) {
      failed = add_frame(&batches[i], recipes[(i + r) % 2], &growing[i], bytes);
      if (!failed && i % 2 == 0)
        add_whole_copy(&batches[i], r, bytes);
      else if (!failed)
        add_copies(&batches[i], r, &growing[i], bytes);
    }
    offsets[i + 1] = offsets[i] + round_up_test(batches[i].targetBytes);
  }
  const size_t total = offsets[WAVES];
  if (!failed) {
    for (size_t i = 0; i < 4 && !failed; ++i) {
      memories[i] = geozl_gpu_alloc(gpu, total, error, sizeof(error));
      failed = memories[i] == NULL;
    }
    got = malloc(total);
    want = calloc(1, total);
    written = calloc(1, total);
    failed = failed || got == NULL || want == NULL || written == NULL;
  }
  for (int round = 0; round < 3 && !failed; ++round) {
    GeozlGpuMemory *memory = memories[round];
    for (size_t i = 0; i < WAVES && !failed; ++i) {
      // Switch streams halfway through the third round.
      void *stream = round == 2 ? streams.stream[i / 3] : NULL;
      const GeozlGpuTarget target = {geozl_gpu_pointer(memory), offsets[i],
                                     batches[i].targetBytes, stream};
      failed = geozl_gpu_decode(gpu, batches[i].frames, batches[i].count,
                                batches[i].copies, batches[i].copyCount, bytes,
                                bytes, &target, error, sizeof(error)) != 0;
      if (!failed && round == 0) {
        expect(&batches[i], bytes, want + offsets[i], written + offsets[i]);
      }
    }
    if (!failed)
      failed = geozl_gpu_synchronize(gpu, error, sizeof(error)) != 0 ||
               geozl_gpu_memory_download(memory, got, total, error,
                                         sizeof(error)) != 0;
    for (size_t i = 0; i < total && !failed; ++i) {
      if (written[i] && got[i] != want[i]) {
        fprintf(stderr, "overlapping calls: byte %zu is %u, want %u\n", i,
                got[i], want[i]);
        failed = 1;
      }
    }
  }
  if (!failed) {
    Batch *bad = &batches[1];
    GeozlGpuMemory *memory = memories[3];
    const GeozlGpuTarget first = {geozl_gpu_pointer(memory), offsets[0],
                                  batches[0].targetBytes, NULL};
    const GeozlGpuTarget second = {geozl_gpu_pointer(memory), offsets[1],
                                   bad->targetBytes, NULL};
    memset(bad->data[0] + bad->frames[0].size / 2, 0xff, 32);
    failed = geozl_gpu_decode(gpu, batches[0].frames, batches[0].count,
                              batches[0].copies, batches[0].copyCount, bytes,
                              bytes, &first, error, sizeof(error)) != 0;
    if (!failed &&
        geozl_gpu_decode(gpu, bad->frames, bad->count, bad->copies,
                         bad->copyCount, bytes, bytes, &second, error,
                         sizeof(error)) == 0) {
      fprintf(stderr, "a corrupted frame was accepted between waves\n");
      failed = 1;
    }
    if (!failed)
      failed = geozl_gpu_synchronize(gpu, error, sizeof(error)) != 0 ||
               geozl_gpu_memory_download(memory, got, total, error,
                                         sizeof(error)) != 0;
    for (size_t i = 0; i < batches[0].targetBytes && !failed; ++i) {
      if (written[i] && got[i] != want[i]) {
        fprintf(stderr, "the wave before a failure: byte %zu differs\n", i);
        failed = 1;
      }
    }
  }
  geozl_gpu_synchronize(gpu, NULL, 0);
  streams_close(&streams);
  for (size_t i = 0; i < 4; ++i)
    geozl_gpu_free(memories[i]);
  free(written);
  free(want);
  free(got);
  for (size_t i = 0; i < WAVES; ++i)
    clear(&batches[i]);
  return failed;
}

int main(void) {
  if (!geozl_gpu_available(DEVICE)) {
    puts("test_gpu_decode: skipped, no GPU backend");
    return 0;
  }
  char error[512];
  GeozlGpu *gpu =
      geozl_gpu_create(DEVICE, 0, 4, 1 << 20, 1, error, sizeof(error));
  if (gpu == NULL) {
    fprintf(stderr, "test_gpu_decode: %s\n", error);
    return 1;
  }
  const int failed = check_mixed(gpu, 2) || check_mixed(gpu, 1) ||
                     check_host_widths(gpu) || check_refusals(gpu) ||
                     check_capture_errors(gpu) ||
                     check_segmented_frames(gpu, 1) ||
                     check_segmented_frames(gpu, 2) ||
                     check_forced_host(gpu) || check_direct(gpu, 1) ||
                     check_direct(gpu, 2) || check_waves(gpu, 1) ||
                     check_waves(gpu, 2) ||
                     check_plane_limits() ||
                     check_batch_limit(gpu);
  geozl_gpu_destroy(gpu);
  if (failed) {
    fprintf(stderr, "test_gpu_decode: failed\n");
    return 1;
  }
  puts("test_gpu_decode: ok");
  return 0;
}
