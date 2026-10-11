#include "gpu/pfor_capture.h"
#include "gpu/pfor_decode.h"

#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int check_wrapped_frame(GeozlPforGpuDecoder *decoder, const void *input,
                               size_t elements, size_t elementBytes,
                               uint32_t width, uint32_t planes) {
  if (elementBytes == 1)
    return 0;
  char error[512];
  geozl_2d_graph *graph = NULL;
  const int dtype = elementBytes == 1 ? GEOZL_DT_U8 : GEOZL_DT_U16;
  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>pfor", width, planes,
                            "LINEAR:MAX_ERROR=2", dtype, GEOZL_NODATA_NONE, 0,
                            input, elements, elementBytes, error,
                            sizeof(error)) != 0)
    return 1;

  const size_t capacity = ZL_compressBound(elements * elementBytes);
  uint8_t *frame = malloc(capacity);
  size_t frameSize;
  if (frame == NULL ||
      geozl_2d_compress_graph_c(graph, input, elements, frame, capacity,
                                &frameSize, error, sizeof(error)) != 0) {
    geozl_2d_graph_close_c(graph);
    free(frame);
    return 1;
  }
  geozl_2d_graph_close_c(graph);

  const void *frames[] = {frame};
  const size_t sizes[] = {frameSize};
  GeozlGpuMemory *memory =
      geozl_pfor_gpu_decode(decoder, frames, sizes, 1, elements * elementBytes,
                            NULL, error, sizeof(error));
  free(frame);
  if (memory != NULL) {
    geozl_gpu_memory_destroy(memory);
    return 1;
  }
  return 0;
}

static int
check_raw_fallback(GeozlPforGpuDecoder *decoder, const void *compressedFrame,
                   size_t compressedFrameSize, const void *compressedOutput,
                   size_t compressedOutputSize, size_t elementBytes) {
  enum { rawElements = 7 };
  const size_t rawBytes = rawElements * elementBytes;
  uint8_t raw[rawElements * 2];
  for (size_t i = 0; i < rawElements; ++i) {
    const uint16_t value = (uint16_t)(i * 97u + 31u);
    if (elementBytes == 1)
      raw[i] = (uint8_t)value;
    else
      memcpy(raw + i * 2, &value, 2);
  }

  char error[512] = {0};
  geozl_2d_graph *graph = NULL;
  const int dtype = elementBytes == 1 ? GEOZL_DT_U8 : GEOZL_DT_U16;
  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>pfor", rawElements, 1, NULL,
                            dtype, GEOZL_NODATA_NONE, 0, raw, rawElements,
                            elementBytes, error, sizeof(error)) != 0)
    return 1;
  const size_t capacity = ZL_compressBound(rawBytes);
  uint8_t *frame = malloc(capacity);
  size_t frameSize = 0;
  const int compressFailed =
      frame == NULL ||
      geozl_2d_compress_graph_c(graph, raw, rawElements, frame, capacity,
                                &frameSize, error, sizeof(error)) != 0;
  geozl_2d_graph_close_c(graph);
  if (compressFailed) {
    free(frame);
    return 1;
  }

  uint8_t *payload = malloc(frameSize + rawBytes + 64);
  uint8_t decoded[rawElements * 2];
  CapturedPforFrame captured;
  PforCapture capture = {0};
  const int isRaw =
      payload != NULL &&
      pfor_capture_init(&capture, payload, frameSize + rawBytes) == 0 &&
      pfor_capture_frame(&capture, &captured, decoded, rawBytes, frame,
                         frameSize) == PFOR_CAPTURE_RAW;
  if (payload != NULL && capture.dctx != NULL)
    pfor_capture_destroy(&capture);
  free(payload);
  if (!isRaw) {
    free(frame);
    return 1;
  }

  const void *frames[] = {compressedFrame, frame};
  const size_t frameSizes[] = {compressedFrameSize, frameSize};
  size_t outputBytes = 0;
  GeozlGpuMemory *memory = geozl_pfor_gpu_decode(
      decoder, frames, frameSizes, 2, compressedOutputSize + rawBytes,
      &outputBytes, error, sizeof(error));
  uint8_t *output = malloc(compressedOutputSize + rawBytes);
  const int failed =
      memory == NULL || output == NULL ||
      outputBytes != compressedOutputSize + rawBytes ||
      geozl_gpu_memory_download(memory, output, outputBytes, error,
                                sizeof(error)) != 0 ||
      memcmp(output, compressedOutput, compressedOutputSize) != 0 ||
      memcmp(output + compressedOutputSize, raw, rawBytes) != 0;
  if (failed)
    fprintf(stderr, "test_pfor_decode: raw fallback: %s\n",
            error[0] ? error : "GPU output mismatch");
  geozl_gpu_memory_destroy(memory);
  free(output);
  free(frame);
  return failed;
}

static int check_id_pfor(GeozlPforGpuDecoder *decoder, size_t elementBytes) {
  enum { count = 4096 };
  const size_t bytes = count * elementBytes;
  uint8_t *input = malloc(bytes);
  if (input == NULL)
    return 1;
  for (size_t i = 0; i < count; ++i) {
    const uint16_t value = (uint16_t)((i * 5u + i / 31u) & 31u);
    if (elementBytes == 1)
      input[i] = (uint8_t)value;
    else
      memcpy(input + i * 2, &value, 2);
  }

  char error[512] = {0};
  geozl_2d_graph *graph = NULL;
  const int dtype = elementBytes == 1 ? GEOZL_DT_U8 : GEOZL_DT_U16;
  if (geozl_2d_graph_open_c(&graph, "id>pfor", count, 1, NULL, dtype,
                            GEOZL_NODATA_NONE, 0, input, count, elementBytes,
                            error, sizeof(error)) != 0) {
    free(input);
    return 1;
  }
  const size_t capacity = ZL_compressBound(bytes);
  uint8_t *frame = malloc(capacity);
  size_t frameSize = 0;
  const int compressFailed =
      frame == NULL ||
      geozl_2d_compress_graph_c(graph, input, count, frame, capacity,
                                &frameSize, error, sizeof(error)) != 0;
  geozl_2d_graph_close_c(graph);
  if (compressFailed) {
    free(frame);
    free(input);
    return 1;
  }

  const void *frames[] = {frame};
  const size_t frameSizes[] = {frameSize};
  size_t outputBytes = 0;
  GeozlGpuMemory *memory =
      geozl_pfor_gpu_decode(decoder, frames, frameSizes, 1, bytes, &outputBytes,
                            error, sizeof(error));
  uint8_t *output = malloc(bytes);
  const int failed = memory == NULL || output == NULL || outputBytes != bytes ||
                     geozl_gpu_memory_download(memory, output, bytes, error,
                                               sizeof(error)) != 0 ||
                     memcmp(output, input, bytes) != 0;
  if (failed)
    fprintf(stderr, "test_pfor_decode: id>pfor: %s\n",
            error[0] ? error : "GPU output mismatch");
  geozl_gpu_memory_destroy(memory);
  free(output);
  free(frame);
  free(input);
  return failed;
}

static int run_case(const char *modulePath, size_t elementBytes) {
  enum { frameCount = 3, width = 300, rows = 3, planes = 2 };
  const size_t elements = width * rows * planes;
  const size_t frameBytes = elements * elementBytes;
  uint8_t *input = malloc(frameCount * frameBytes);
  if (input == NULL)
    return 1;
  for (size_t frame = 0; frame < frameCount; ++frame) {
    for (size_t i = 0; i < elements; ++i) {
      const uint16_t value = (uint16_t)(frame * 113 + i * 7 + i / width);
      if (elementBytes == 1)
        input[frame * frameBytes + i] = (uint8_t)value;
      else
        memcpy(input + frame * frameBytes + i * 2, &value, 2);
    }
  }

  char error[512];
  geozl_2d_graph *graph = NULL;
  const int dtype = elementBytes == 1 ? GEOZL_DT_U8 : GEOZL_DT_U16;
  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>pfor", width, planes, NULL,
                            dtype, GEOZL_NODATA_NONE, 0, input, elements,
                            elementBytes, error, sizeof(error)) != 0) {
    fprintf(stderr, "test_pfor_decode: %s\n", error);
    free(input);
    return 1;
  }

  const size_t frameCapacity = ZL_compressBound(frameBytes);
  uint8_t *storage = malloc(frameCount * frameCapacity);
  const void *frames[frameCount];
  size_t frameSizes[frameCount];
  if (storage == NULL) {
    geozl_2d_graph_close_c(graph);
    free(input);
    return 1;
  }
  for (size_t frame = 0; frame < frameCount; ++frame) {
    frames[frame] = storage + frame * frameCapacity;
    if (geozl_2d_compress_graph_c(graph, input + frame * frameBytes, elements,
                                  storage + frame * frameCapacity,
                                  frameCapacity, &frameSizes[frame], error,
                                  sizeof(error)) != 0) {
      fprintf(stderr, "test_pfor_decode: %s\n", error);
      geozl_2d_graph_close_c(graph);
      free(storage);
      free(input);
      return 1;
    }
  }
  geozl_2d_graph_close_c(graph);

  size_t moduleSize;
  void *module = read_file(modulePath, &moduleSize);
  if (module == NULL) {
    fprintf(stderr, "test_pfor_decode: cannot read %s\n", modulePath);
    free(storage);
    free(input);
    return 1;
  }
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  GeozlPforGpuDecoder *decoder = geozl_pfor_gpu_decoder_create(
      module, moduleSize, elementBytes, device, error, sizeof(error));
  free(module);
  if (decoder == NULL) {
    fprintf(stderr, "test_pfor_decode: %s\n", error);
    free(storage);
    free(input);
    return 1;
  }

  size_t outputBytes;
  GeozlGpuMemory *memory = geozl_pfor_gpu_decode(
      decoder, frames, frameSizes, frameCount, frameCount * frameBytes,
      &outputBytes, error, sizeof(error));
  GeozlGpuMemory *tooLarge = geozl_pfor_gpu_decode(
      decoder, frames, frameSizes, frameCount, frameCount * frameBytes - 1,
      NULL, error, sizeof(error));
  if (tooLarge != NULL) {
    geozl_gpu_memory_destroy(tooLarge);
    geozl_gpu_memory_destroy(memory);
    geozl_pfor_gpu_decoder_destroy(decoder);
    free(storage);
    free(input);
    return 1;
  }
  error[0] = '\0';
  const int acceptedWrapped = check_wrapped_frame(decoder, input, elements,
                                                  elementBytes, width, planes);
  const int rawFailed = check_raw_fallback(decoder, frames[0], frameSizes[0],
                                           input, frameBytes, elementBytes);
  const int idFailed = check_id_pfor(decoder, elementBytes);
  geozl_pfor_gpu_decoder_destroy(decoder);
  if (memory == NULL || acceptedWrapped || rawFailed || idFailed) {
    fprintf(stderr, "test_pfor_decode: %s\n",
            acceptedWrapped ? "accepted a wrapped PFOR frame"
            : rawFailed     ? "raw frame fallback failed"
            : idFailed      ? "id>pfor failed"
                            : error);
    geozl_gpu_memory_destroy(memory);
    free(storage);
    free(input);
    return 1;
  }

  uint8_t *output = malloc(frameCount * frameBytes);
  const int failed =
      output == NULL || outputBytes != frameCount * frameBytes ||
      geozl_gpu_memory_download(memory, output, frameCount * frameBytes, error,
                                sizeof(error)) != 0 ||
      memcmp(output, input, frameCount * frameBytes) != 0;
  if (failed)
    fprintf(stderr, "test_pfor_decode: %s\n",
            error[0] ? error : "GPU output mismatch");

  geozl_gpu_memory_destroy(memory);
  free(output);
  free(storage);
  free(input);
  return failed;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: test_pfor_decode pzp1.metal pzp2.metal\n");
    return 2;
  }
  if (run_case(argv[1], 1) || run_case(argv[2], 2))
    return 1;
  puts("test_pfor_decode: ok");
  return 0;
}
