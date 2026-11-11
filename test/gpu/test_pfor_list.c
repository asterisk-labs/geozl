#include "gpu/pfor_plan.h"
#include "gpu/runtime.h"

#include "pfor/pfor_check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_pfor_list: %s\n", message);
  return 1;
}

static void *read_file(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (file == NULL || fseek(file, 0, SEEK_END) != 0)
    return NULL;
  const long length = ftell(file);
  if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return NULL;
  }
  void *data = malloc((size_t)length);
  if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
    free(data);
    fclose(file);
    return NULL;
  }
  fclose(file);
  *size = (size_t)length;
  return data;
}

static void put_bits(uint8_t *dst, size_t *bit, unsigned width,
                     unsigned value) {
  for (unsigned i = 0; i < width; ++i, ++*bit)
    dst[*bit >> 3] |= (uint8_t)(((value >> i) & 1u) << (*bit & 7));
}

enum { exceptions = 81, exceptionBits = 7, frameLimit = 5, longBlocks = 300 };

static size_t write_block(uint8_t *dst) {
  const size_t highBytes = (exceptions * exceptionBits + 7) / 8;
  dst[0] = 1;
  dst[1] = exceptions;
  dst[2] = 0x80u | exceptionBits;
  for (unsigned i = 0; i < exceptions; ++i)
    dst[3 + i] = (uint8_t)i;
  size_t bit = 0;
  for (unsigned i = 0; i < exceptions; ++i)
    put_bits(dst + 3 + exceptions, &bit, exceptionBits, 1);
  return 3 + exceptions + highBytes + geozl_pfor_body_bytes(1);
}

typedef struct {
  uint16_t output[longBlocks * GEOZL_PFOR_BLOCK];
  uint32_t status[2];
} Result;

static unsigned sample(const Result *result, size_t elementBytes, size_t i) {
  return elementBytes == 1 ? ((const uint8_t *)result->output)[i]
                           : result->output[i];
}

static PforGpuFrame row_frame(size_t payloadOffset, size_t payloadSize,
                              size_t outputBase, int planar) {
  return (PforGpuFrame){{outputBase, GEOZL_PFOR_BLOCK, GEOZL_PFOR_BLOCK, 1},
                        payloadOffset,
                        payloadSize,
                        planar ? PFOR_GPU_PLANAR_ZIGZAG : PFOR_GPU_ID};
}

static GeozlGpuBuffer input(void *host, size_t count, size_t elementBytes,
                            void *dummy) {
  return count != 0
             ? (GeozlGpuBuffer){host, count * elementBytes, elementBytes,
                                GEOZL_GPU_COPY_IN, -1}
             : (GeozlGpuBuffer){dummy, 4 * sizeof(uint32_t),
                                4 * sizeof(uint32_t), GEOZL_GPU_COPY_IN, -1};
}

static int run(GeozlGpuProgram *program, size_t elementBytes, uint8_t *payload,
               size_t payloadSize, const PforGpuFrame frames[],
               size_t frameCount, size_t samples, Result *result, char *error,
               size_t errorCapacity) {
  PforGpuPlan plan = {0};
  PforGpuBandPlan bands = {0};
  PforGpuFrame idFrames[frameLimit];
  size_t idCount = 0;
  for (size_t i = 0; i < frameCount; ++i)
    if (frames[i].transform == PFOR_GPU_ID)
      idFrames[idCount++] = frames[i];
  if ((idCount != 0 && pfor_gpu_plan_build(&plan, idFrames, idCount, payload,
                                           payloadSize, elementBytes) != 0) ||
      pfor_gpu_band_plan_build(&bands, frames, frameCount, NULL, NULL) != 0) {
    snprintf(error, errorCapacity, "planner rejected the test frames");
    pfor_gpu_plan_free(&plan);
    return 1;
  }

  uint32_t dummy[4] = {0};
  result->status[PFOR_GPU_STATUS_HEADER] = UINT32_MAX;
  result->status[PFOR_GPU_STATUS_POSITIONS] = UINT32_MAX;
  const size_t outputBytes = samples * elementBytes;
  const size_t blockRefs = bands.blockCount == 0 ? 1 : bands.blockCount;
  GeozlGpuBuffer buffers[] = {
      {payload, payloadSize + 64, 1, GEOZL_GPU_COPY_IN, -1},
      input(plan.blocks, plan.blockCount, sizeof(*plan.blocks), dummy),
      {NULL, outputBytes, elementBytes, GEOZL_GPU_RESULT, -1},
      {NULL, outputBytes, 1, 0, 2},
      input(bands.frames, bands.frameCount, sizeof(*bands.frames), dummy),
      input(bands.strips, bands.stripCount, sizeof(*bands.strips), dummy),
      {NULL, blockRefs * sizeof(GpuPforBlockRef), sizeof(GpuPforBlockRef), 0,
       -1},
      {result->status, sizeof(result->status), sizeof(uint32_t),
       GEOZL_GPU_COPY_IN | GEOZL_GPU_COPY_OUT, -1},
      input(bands.chunks, bands.chunkCount, sizeof(*bands.chunks), dummy),
      {NULL, (bands.chunkCount == 0 ? 1 : bands.chunkCount) * sizeof(uint32_t),
       sizeof(uint32_t), 0, -1},
      {dummy, elementBytes, elementBytes, GEOZL_GPU_COPY_IN, -1},
      {dummy, 12, 12, GEOZL_GPU_COPY_IN, -1},
  };
  GeozlGpuDispatch dispatches[4];
  size_t dispatchCount = 0;
  if (plan.blockCount != 0)
    dispatches[dispatchCount++] =
        (GeozlGpuDispatch){0, (unsigned)(plan.blockCount / 4), 1, 128};
  if (bands.frameCount != 0) {
    dispatches[dispatchCount++] =
        (GeozlGpuDispatch){1, (unsigned)(bands.chunkCount / 4), 1, 128};
    dispatches[dispatchCount++] =
        (GeozlGpuDispatch){2, (unsigned)(bands.chunkCount / 4), 1, 128};
    dispatches[dispatchCount++] =
        (GeozlGpuDispatch){3, (unsigned)(bands.stripCount / 4), 1, 128};
  }
  GeozlGpuBatch *batch = geozl_gpu_batch_create(
      program, buffers, sizeof(buffers) / sizeof(buffers[0]), error,
      errorCapacity);
  GeozlGpuRunTimes times;
  int failed = batch == NULL ||
               geozl_gpu_batch_run(batch, dispatches, dispatchCount, &times,
                                   error, errorCapacity) != 0 ||
               geozl_gpu_batch_download(batch, error, errorCapacity) != 0;
  // A dispatch the device rejects must leave the finished result takeable.
  if (!failed) {
    const GeozlGpuDispatch tooWide = {0, 1, 1, 4096};
    if (geozl_gpu_batch_run(batch, &tooWide, 1, &times, error, errorCapacity) ==
        0) {
      snprintf(error, errorCapacity, "a 4096-thread group was accepted");
      failed = 1;
    } else {
      error[0] = '\0';
    }
  }
  if (!failed && strcmp(geozl_gpu_program_backend(program), "cuda") == 0) {
    const GeozlGpuDispatch tooTall = {0, 1, 65536, 128};
    if (geozl_gpu_batch_run(batch, &tooTall, 1, &times, error, errorCapacity) ==
            0 ||
        strstr(error, "exceeds device limits") == NULL) {
      snprintf(error, errorCapacity, "CUDA accepted a grid taller than 65535");
      failed = 1;
    } else {
      error[0] = '\0';
    }
  }
  GeozlGpuMemory *memory =
      failed ? NULL : geozl_gpu_batch_take(batch, 2, error, errorCapacity);
  geozl_gpu_batch_destroy(batch);
  failed = memory == NULL ||
           geozl_gpu_memory_download(memory, result->output, outputBytes, error,
                                     errorCapacity) != 0;
  geozl_gpu_memory_destroy(memory);
  pfor_gpu_band_plan_free(&bands);
  pfor_gpu_plan_free(&plan);
  return failed;
}

static unsigned summed(size_t i) {
  return i < exceptions ? (unsigned)i + 1 : exceptions;
}

static int check_chunks(GeozlGpuProgram *program, size_t elementBytes,
                        size_t blockSize, Result *result, char *error,
                        size_t errorCapacity) {
  static uint8_t payload[longBlocks * 256 + 64];
  uint8_t block[256] = {0};
  write_block(block);
  memset(payload, 0, sizeof(payload));
  for (size_t k = 0; k < longBlocks; ++k) {
    memcpy(payload + 3 * k, block, 3);
    memcpy(payload + 3 * longBlocks + k * (blockSize - 3), block + 3,
           blockSize - 3);
  }
  const size_t samples = longBlocks * GEOZL_PFOR_BLOCK;
  const size_t payloadSize = longBlocks * blockSize;
  PforGpuFrame frame = row_frame(0, payloadSize, 0, 1);
  frame.shape.nbElts = samples;
  frame.shape.width = samples;
  if (run(program, elementBytes, payload, payloadSize, &frame, 1, samples,
          result, error, errorCapacity))
    return 1;
  const unsigned mask = elementBytes == 1 ? 0xFFu : 0xFFFFu;
  unsigned sum = 0;
  for (size_t x = 0; x < samples; ++x) {
    if (x % 1024 == 0)
      sum = 0;
    sum += x % GEOZL_PFOR_BLOCK < exceptions;
    if (sample(result, elementBytes, x) != (sum & mask)) {
      snprintf(error, errorCapacity, "chunked frame differs at sample %zu", x);
      return 1;
    }
  }
  if (result->status[PFOR_GPU_STATUS_HEADER] != UINT32_MAX ||
      result->status[PFOR_GPU_STATUS_POSITIONS] != UINT32_MAX) {
    snprintf(error, errorCapacity, "a chunked frame set status");
    return 1;
  }
  for (int forgery = 0; forgery < 2; ++forgery) {
    PforGpuFrame forged = frame;
    if (forgery == 0)
      payload[3 * 280] = 17;
    else
      forged.payloadSize += 1;
    if (run(program, elementBytes, payload, forged.payloadSize, &forged, 1,
            samples, result, error, errorCapacity))
      return 1;
    payload[3 * 280] = block[0];
    if (result->status[PFOR_GPU_STATUS_HEADER] != 0) {
      snprintf(error, errorCapacity, "chunked forgery %d was accepted",
               forgery);
      return 1;
    }
  }
  return 0;
}

static int check_width(const char *modulePath, size_t elementBytes) {
  size_t moduleSize;
  void *module = read_file(modulePath, &moduleSize);
  char error[512] = {0};
  static const char *const kernels[] = {"unpack_blocks", "size_chunks",
                                        "plan_blocks", "decode_strips"};
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  GeozlGpuProgram *program =
      module == NULL
          ? NULL
          : geozl_gpu_program_create(GPU_BACKEND, module, moduleSize, device,
                                     kernels, 4, error, sizeof(error));
  free(module);
  if (program == NULL)
    return fail(module == NULL ? "could not read the shader" : error);

  uint8_t payload[frameLimit * 256 + 64] = {0};
  const size_t blockSize = write_block(payload);
  memcpy(payload + blockSize, payload, blockSize);
  const PforGpuFrame good[] = {
      row_frame(0, blockSize, 0, 0),
      row_frame(blockSize, blockSize, GEOZL_PFOR_BLOCK, 1),
  };
  static Result result;
  int failed = run(program, elementBytes, payload, 2 * blockSize, good, 2,
                   2 * GEOZL_PFOR_BLOCK, &result, error, sizeof(error));
  for (size_t i = 0; i < GEOZL_PFOR_BLOCK && !failed; ++i) {
    if (sample(&result, elementBytes, i) != (i < exceptions ? 2u : 0u) ||
        sample(&result, elementBytes, GEOZL_PFOR_BLOCK + i) != summed(i)) {
      snprintf(error, sizeof(error), "GPU output mismatch at sample %zu", i);
      failed = 1;
    }
  }
  if (!failed && (result.status[PFOR_GPU_STATUS_HEADER] != UINT32_MAX ||
                  result.status[PFOR_GPU_STATUS_POSITIONS] != UINT32_MAX)) {
    snprintf(error, sizeof(error), "a valid list set status %u %u",
             result.status[0], result.status[1]);
    failed = 1;
  }

  if (!failed) {
    for (size_t k = 2; k < frameLimit; ++k)
      memcpy(payload + k * blockSize, payload, blockSize);
    payload[blockSize + 3 + 6] = payload[blockSize + 3 + 5];
    payload[2 * blockSize] = 17;
    const PforGpuFrame bad[] = {
        row_frame(0, blockSize, 0, 1),
        row_frame(blockSize, blockSize, GEOZL_PFOR_BLOCK, 1),
        row_frame(2 * blockSize, blockSize, 2 * GEOZL_PFOR_BLOCK, 1),
        row_frame(3 * blockSize, blockSize - 1, 3 * GEOZL_PFOR_BLOCK, 1),
        row_frame(4 * blockSize, blockSize + 1, 4 * GEOZL_PFOR_BLOCK, 1),
    };
    failed = run(program, elementBytes, payload, 5 * blockSize + 1, bad, 5,
                 frameLimit * GEOZL_PFOR_BLOCK, &result, error, sizeof(error));
    if (!failed && (result.status[PFOR_GPU_STATUS_HEADER] != 2 ||
                    result.status[PFOR_GPU_STATUS_POSITIONS] != 1)) {
      snprintf(error, sizeof(error), "forged frames set status %u %u",
               result.status[0], result.status[1]);
      failed = 1;
    }
    for (size_t i = 0; i < GEOZL_PFOR_BLOCK && !failed; ++i) {
      if (sample(&result, elementBytes, i) != summed(i) ||
          sample(&result, elementBytes, 2 * GEOZL_PFOR_BLOCK + i) != 0 ||
          sample(&result, elementBytes, 3 * GEOZL_PFOR_BLOCK + i) != 0) {
        snprintf(error, sizeof(error), "forged frames changed sample %zu", i);
        failed = 1;
      }
    }
  }
  if (!failed)
    failed = check_chunks(program, elementBytes, blockSize, &result, error,
                          sizeof(error));
  geozl_gpu_program_destroy(program);
  if (failed) {
    fprintf(stderr, "test_pfor_list: %zu-byte elements: %s\n", elementBytes,
            error[0] ? error : "GPU decode failed");
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: test_pfor_list pzp1.metal|ptx pzp2.metal|ptx\n");
    return 2;
  }
  if (check_width(argv[1], 1) || check_width(argv[2], 2))
    return 1;
  puts("test_pfor_list: ok");
  return 0;
}
