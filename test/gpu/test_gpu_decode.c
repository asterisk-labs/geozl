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
#include "transpose_pivco/graph_transpose_pivco.h"

#include "openzl/zl_compress.h"

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
static int run(GeozlGpu *gpu, const Batch *batch, size_t sampleBytes,
               size_t elementBytes, const char *name) {
  char error[512];
  const size_t shift = 64 * sampleBytes;
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
    if (geozl_gpu_decode(gpu, batch->frames, batch->count, batch->copies,
                         batch->copyCount, sampleBytes, elementBytes, &target,
                         error, sizeof(error)) != 0 ||
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
                     check_capture_errors(gpu) || check_batch_limit(gpu);
  geozl_gpu_destroy(gpu);
  if (failed) {
    fprintf(stderr, "test_gpu_decode: failed\n");
    return 1;
  }
  puts("test_gpu_decode: ok");
  return 0;
}
