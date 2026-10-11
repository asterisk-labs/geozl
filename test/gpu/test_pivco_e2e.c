#include "gpu/pivco_capture.h"
#include "gpu/pivco_decode.h"
#include "gpu/pivco_plan.h"

#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint8_t *data;
  size_t size;
} Frame;

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

static int compress_frame(Frame *frame, const void *input, size_t elements,
                          size_t elementBytes, uint32_t width, uint32_t planes,
                          char *error, size_t errorCapacity) {
  geozl_2d_graph *graph = NULL;
  const int dtype = elementBytes == 1 ? GEOZL_DT_U8 : GEOZL_DT_U16;
  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>transpose>pivco", width,
                            planes, NULL, dtype, GEOZL_NODATA_NONE, 0, input,
                            elements, elementBytes, error, errorCapacity) != 0)
    return 1;
  const size_t capacity = ZL_compressBound(elements * elementBytes);
  frame->data = malloc(capacity);
  const int failed =
      frame->data == NULL ||
      geozl_2d_compress_graph_c(graph, input, elements, frame->data, capacity,
                                &frame->size, error, errorCapacity) != 0;
  geozl_2d_graph_close_c(graph);
  return failed;
}

static PivcoCaptureResult capture_one(const Frame *frame, void *output,
                                      size_t outputBytes,
                                      CapturedFrame *captured,
                                      uint8_t **payloadOut,
                                      size_t *payloadSizeOut) {
  const size_t capacity = frame->size + outputBytes + 64;
  uint8_t *payload = malloc(capacity);
  PivcoCapture capture = {0};
  if (payload == NULL || pivco_capture_init(&capture, payload, capacity) != 0) {
    free(payload);
    return PIVCO_CAPTURE_ERROR;
  }
  const PivcoCaptureResult result = pivco_capture_frame(
      &capture, captured, output, outputBytes, frame->data, frame->size);
  *payloadSizeOut = capture.payloadSize;
  pivco_capture_destroy(&capture);
  *payloadOut = payload;
  return result;
}

static uint8_t skewed_symbol(size_t value) {
  const unsigned bucket = (unsigned)(value % 10);
  return bucket < 4 ? 17 : bucket < 7 ? 63 : bucket < 9 ? 129 : 241;
}

static int check_id_pivco(GeozlPivcoGpuDecoder *decoder, size_t elementBytes,
                          int dominant) {
  enum { count = 65536 };
  const size_t bytes = count * elementBytes;
  uint8_t *input = malloc(bytes);
  if (input == NULL)
    return 1;
  for (size_t i = 0; i < count; ++i) {
    const uint16_t value =
        dominant && i % 20 != 0
            ? UINT16_C(0x813F)
            : (uint16_t)skewed_symbol(i) |
                  ((uint16_t)skewed_symbol(i / 10 + i / 97) << 8);
    if (elementBytes == 1)
      input[i] = (uint8_t)value;
    else
      memcpy(input + i * 2, &value, 2);
  }

  const char *recipe = elementBytes == 1 ? "id>pivco" : "id>transpose>pivco";
  const int dtype = elementBytes == 1 ? GEOZL_DT_U8 : GEOZL_DT_U16;
  char error[512] = {0};
  geozl_2d_graph *graph = NULL;
  if (geozl_2d_graph_open_c(&graph, recipe, count, 1, NULL, dtype,
                            GEOZL_NODATA_NONE, 0, input, count, elementBytes,
                            error, sizeof(error)) != 0) {
    free(input);
    return 1;
  }
  Frame frame = {0};
  const size_t capacity = ZL_compressBound(bytes);
  frame.data = malloc(capacity);
  const int compressFailed =
      frame.data == NULL ||
      geozl_2d_compress_graph_c(graph, input, count, frame.data, capacity,
                                &frame.size, error, sizeof(error)) != 0;
  geozl_2d_graph_close_c(graph);
  if (compressFailed) {
    free(frame.data);
    free(input);
    return 1;
  }
  uint8_t *captureOutput = malloc(bytes);
  uint8_t *payload = NULL;
  size_t payloadSize = 0;
  CapturedFrame captured;
  const PivcoCaptureResult captureResult = capture_one(
      &frame, captureOutput, bytes, &captured, &payload, &payloadSize);
  const int wrongPath = captureResult != PIVCO_CAPTURE_CODEC ||
                        captured.kind != PIVCO_CAPTURE_KIND_ID;
  free(payload);
  free(captureOutput);
  if (wrongPath) {
    fprintf(stderr, "test_pivco_e2e: %s%s used capture path %d/%d\n", recipe,
            dominant ? " with a dominant value" : "", captureResult,
            captured.kind);
    free(frame.data);
    free(input);
    return 1;
  }

  const void *frames[] = {frame.data};
  const size_t frameSizes[] = {frame.size};
  size_t outputBytes = 0;
  GeozlGpuMemory *memory =
      geozl_pivco_gpu_decode(decoder, frames, frameSizes, 1, bytes,
                             &outputBytes, error, sizeof(error));
  uint8_t *output = malloc(bytes);
  const int failed = memory == NULL || output == NULL || outputBytes != bytes ||
                     geozl_gpu_memory_download(memory, output, bytes, error,
                                               sizeof(error)) != 0 ||
                     memcmp(output, input, bytes) != 0;
  if (failed)
    fprintf(stderr, "test_pivco_e2e: %s%s: %s\n", recipe,
            dominant ? " with a dominant value" : "",
            error[0] ? error : "GPU output mismatch");
  geozl_gpu_memory_destroy(memory);
  free(output);
  free(frame.data);
  free(input);
  return failed;
}

static int run_case(const char *modulePath, size_t elementBytes) {
  enum {
    regularWidth = 300,
    regularRows = 3,
    regularPlanes = 2,
    regularElements = regularWidth * regularRows * regularPlanes,
    constantElements = 256 * 256,
    rawElements = 7,
  };
  const size_t regularBytes = regularElements * elementBytes;
  const size_t constantBytes = constantElements * elementBytes;
  const size_t rawBytes = rawElements * elementBytes;
  const size_t totalBytes = regularBytes + constantBytes + rawBytes;
  uint8_t *expected = malloc(totalBytes);
  if (expected == NULL)
    return 1;

  for (size_t i = 0; i < regularElements; ++i) {
    const uint16_t value = (uint16_t)(i * 7u + i / regularWidth);
    if (elementBytes == 1)
      expected[i] = (uint8_t)value;
    else
      memcpy(expected + i * 2, &value, 2);
  }
  const uint16_t constantValue = elementBytes == 1 ? 123 : 1234;
  for (size_t i = 0; i < constantElements; ++i)
    memcpy(expected + regularBytes + i * elementBytes, &constantValue,
           elementBytes);
  for (size_t i = 0; i < rawElements; ++i) {
    const uint16_t value = (uint16_t)(i * 97u + 31u);
    uint8_t *dst = expected + regularBytes + constantBytes + i * elementBytes;
    if (elementBytes == 1)
      *dst = (uint8_t)value;
    else
      memcpy(dst, &value, 2);
  }

  char error[512] = {0};
  Frame encoded[3] = {0};
  if (compress_frame(&encoded[0], expected, regularElements, elementBytes,
                     regularWidth, regularPlanes, error, sizeof(error)) ||
      compress_frame(&encoded[1], expected + regularBytes, constantElements,
                     elementBytes, 256, 1, error, sizeof(error)) ||
      compress_frame(&encoded[2], expected + regularBytes + constantBytes,
                     rawElements, elementBytes, rawElements, 1, error,
                     sizeof(error))) {
    fprintf(stderr, "test_pivco_e2e: %s\n", error);
    goto fail;
  }

  uint8_t *captureOutput =
      malloc(constantBytes > rawBytes ? constantBytes : rawBytes);
  uint8_t *constantPayload = NULL;
  size_t constantPayloadSize = 0;
  CapturedFrame constantFrame;
  const PivcoCaptureResult constantResult =
      capture_one(&encoded[1], captureOutput, constantBytes, &constantFrame,
                  &constantPayload, &constantPayloadSize);
  const GeozlGpuFrameShape constantShape = {0, constantElements, 256, 1};
  PivcoGpuPlan constantPlan = {0};
  const int planFailed =
      constantResult == PIVCO_CAPTURE_CODEC &&
      pivco_gpu_plan_build(&constantPlan, &constantFrame, &constantShape, 1,
                           constantPayload, constantPayloadSize,
                           elementBytes) != 0;
  const int constantFailed = constantResult != PIVCO_CAPTURE_CODEC ||
                             planFailed || constantPlan.pivcoLaneCount != 0 ||
                             constantPlan.symbolCount != 0;
  const size_t constantPivcoLanes = constantPlan.pivcoLaneCount;
  const size_t constantSymbols = constantPlan.symbolCount;
  pivco_gpu_plan_free(&constantPlan);
  free(constantPayload);

  uint8_t *rawPayload = NULL;
  size_t rawPayloadSize = 0;
  CapturedFrame rawFrame;
  const PivcoCaptureResult rawResult =
      capture_one(&encoded[2], captureOutput, rawBytes, &rawFrame, &rawPayload,
                  &rawPayloadSize);
  free(rawPayload);
  free(captureOutput);
  if (constantFailed || rawResult != PIVCO_CAPTURE_RAW) {
    fprintf(stderr,
            "test_pivco_e2e: expected paths missing "
            "(constant=%d plan=%d lanes=%zu symbols=%zu raw=%d)\n",
            constantResult, planFailed, constantPivcoLanes, constantSymbols,
            rawResult);
    goto fail;
  }

  size_t moduleSize = 0;
  void *module = read_file(modulePath, &moduleSize);
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  GeozlPivcoGpuDecoder *decoder =
      module == NULL
          ? NULL
          : geozl_pivco_gpu_decoder_create(module, moduleSize, elementBytes,
                                           device, error, sizeof(error));
  free(module);
  if (decoder == NULL) {
    fprintf(stderr, "test_pivco_e2e: %s\n",
            module == NULL ? "cannot read shader" : error);
    goto fail;
  }

  const void *frames[] = {encoded[0].data, encoded[1].data, encoded[2].data};
  const size_t frameSizes[] = {encoded[0].size, encoded[1].size,
                               encoded[2].size};
  size_t outputBytes = 0;
  GeozlGpuMemory *memory =
      geozl_pivco_gpu_decode(decoder, frames, frameSizes, 3, totalBytes,
                             &outputBytes, error, sizeof(error));
  uint8_t *output = malloc(totalBytes);
  const int failed = memory == NULL || output == NULL ||
                     outputBytes != totalBytes ||
                     geozl_gpu_memory_download(memory, output, totalBytes,
                                               error, sizeof(error)) != 0 ||
                     memcmp(output, expected, totalBytes) != 0;
  if (failed) {
    size_t mismatch = 0;
    if (output != NULL && memory != NULL && error[0] == '\0') {
      while (mismatch < totalBytes && output[mismatch] == expected[mismatch])
        ++mismatch;
    }
    fprintf(stderr, "test_pivco_e2e: %s at byte %zu\n",
            error[0] ? error : "GPU output mismatch", mismatch);
    if (output != NULL && mismatch < totalBytes)
      fprintf(stderr, "test_pivco_e2e: got %u, expected %u\n", output[mismatch],
              expected[mismatch]);
  }
  geozl_gpu_memory_destroy(memory);
  const int idFailed = check_id_pivco(decoder, elementBytes, 0) ||
                       check_id_pivco(decoder, elementBytes, 1);
  geozl_pivco_gpu_decoder_destroy(decoder);
  free(output);
  for (size_t i = 0; i < 3; ++i)
    free(encoded[i].data);
  free(expected);
  return failed || idFailed;

fail:
  for (size_t i = 0; i < 3; ++i)
    free(encoded[i].data);
  free(expected);
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: test_pivco_e2e pzv1.metal|ptx pzv2.metal|ptx\n");
    return 2;
  }
  if (run_case(argv[1], 1) || run_case(argv[2], 2))
    return 1;
  puts("test_pivco_e2e: ok");
  return 0;
}
