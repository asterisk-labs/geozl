#include "gpu/pivco_capture.h"

#include "capture_frame_builder.h"

#include "common/endian.h"
#include "geozl/dtype.h"
#include "geozl/geozl.h"
#include "pivco_lanes/pivco_lanes.h"
#include "transpose_pivco/graph_transpose_pivco.h"

#include "openzl/zl_compress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_pivco_capture: %s\n", message);
  return 1;
}

static int check_inconsistent_count(PivcoCapture *capture,
                                    CapturedFrame *captured, void *output,
                                    size_t outputSize, uint8_t *frame,
                                    size_t frameCapacity) {
  uint8_t header[TRANSPOSE_PIVCO_HEADER_FIXED + GEOZL_PIVCO_LANE_HEADER_SIZE] =
      {0};
  geozl_st_le64(header, 1024);
  header[8] = 1;
  geozl_st_le32(header + 9, 32768);
  header[13] = GEOZL_PIVCO_LANE_PIVCO;
  geozl_st_le16(header + 14, 1);
  const CaptureFramePlan plan = {
      .ctid = GEOZL_CTID_TRANSPOSE_PIVCO,
      .header = header,
      .headerSize = sizeof(header),
      .outputCount = 2,
      .numericOutputs = 1,
      .outputBytes = {1, 0},
  };
  const size_t frameSize =
      capture_test_build_frame(frame, frameCapacity, outputSize, &plan);
  if (frameSize == 0)
    return fail("could not build an inconsistent frame");
  if (pivco_capture_frame(capture, captured, output, outputSize, frame,
                          frameSize) != PIVCO_CAPTURE_ERROR ||
      captured->seen != -1 || captured->headerSize != 0 ||
      capture->payloadSize != 0)
    return fail("an inconsistent output size reached capture");
  return 0;
}

static int check_transpose_codec(const char *method, PivcoCaptureResult want) {
  enum { count = 4096 };
  uint16_t input[count];
  for (size_t i = 0; i < count; ++i)
    input[i] = (uint16_t)(i % 8) | (uint16_t)((i / 7 % 8) << 8);

  char error[512] = {0};
  geozl_2d_graph *graph = NULL;
  if (geozl_2d_graph_open_c(&graph, method, count, 1, NULL, GEOZL_DT_U16,
                            GEOZL_NODATA_NONE, 0, input, count,
                            sizeof(input[0]), error, sizeof(error)) != 0)
    return fail(error);

  const size_t capacity = ZL_compressBound(sizeof(input));
  uint8_t *frame = malloc(capacity);
  uint8_t *payload = malloc(capacity + 64);
  size_t frameSize = 0;
  const int compressFailed =
      frame == NULL || payload == NULL ||
      geozl_2d_compress_graph_c(graph, input, count, frame, capacity,
                                &frameSize, error, sizeof(error)) != 0;
  geozl_2d_graph_close_c(graph);
  if (compressFailed) {
    free(payload);
    free(frame);
    return fail(error[0] ? error : "allocation failed");
  }

  uint16_t output[count];
  memset(output, 0xA5, sizeof(output));
  PivcoCapture capture;
  CapturedFrame captured;
  if (pivco_capture_init(&capture, payload, capacity + 64) != 0) {
    free(payload);
    free(frame);
    return fail("could not create capture context");
  }
  const PivcoCaptureResult got = pivco_capture_frame(
      &capture, &captured, output, sizeof(output), frame, frameSize);
  const int failed = got != want || (want == PIVCO_CAPTURE_CODEC &&
                                     captured.kind != PIVCO_CAPTURE_KIND_ID);
  pivco_capture_destroy(&capture);
  free(payload);
  free(frame);
  return failed ? fail("transpose PivCo took the wrong capture path") : 0;
}

int main(void) {
  enum { width = 8, planes = 2, count = width * 8 * planes };
  uint16_t input[count];
  for (size_t i = 0; i < count; ++i)
    input[i] = (uint16_t)(i * 17 + i / width);

  char error[512];
  geozl_2d_graph *graph = NULL;
  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>transpose>pivco", width,
                            planes, NULL, GEOZL_DT_U16, GEOZL_NODATA_NONE, 0,
                            input, count, sizeof(input[0]), error,
                            sizeof(error)) != 0) {
    fprintf(stderr, "test_pivco_capture: %s\n", error);
    return 1;
  }

  const size_t frameCapacity = ZL_compressBound(sizeof(input));
  uint8_t *frame = malloc(frameCapacity);
  uint8_t *inconsistentFrame = malloc(frameCapacity);
  uint8_t *payload = malloc(frameCapacity + 64);
  if (frame == NULL || inconsistentFrame == NULL || payload == NULL)
    return fail("allocation failed");
  size_t frameSize = 0;
  if (geozl_2d_compress_graph_c(graph, input, count, frame, frameCapacity,
                                &frameSize, error, sizeof(error)) != 0) {
    fprintf(stderr, "test_pivco_capture: %s\n", error);
    return 1;
  }
  geozl_2d_graph_close_c(graph);

  uint16_t output[count];
  memset(output, 0xA5, sizeof(output));
  PivcoCapture capture;
  CapturedFrame captured;
  if (pivco_capture_init(&capture, payload, frameCapacity + 64) != 0)
    return fail("could not create capture context");
  if (check_inconsistent_count(&capture, &captured, output, 64,
                               inconsistentFrame, frameCapacity) != 0)
    return 1;
  if (pivco_capture_frame(&capture, &captured, output, sizeof(output), frame,
                          frameSize) != 0)
    return fail("valid frame was not captured");
  for (size_t i = 0; i < sizeof(output); ++i) {
    if (((const uint8_t *)output)[i] != 0xA5)
      return fail("capture wrote the output buffer");
  }
  if (captured.headerSize == 0 || captured.bitsSize == 0 ||
      capture.payloadSize < captured.bitsSize)
    return fail("captured inputs are incomplete");

  frame[frameSize - 2] ^= 1;
  if (pivco_capture_frame(&capture, &captured, output, sizeof(output), frame,
                          frameSize) == 0 ||
      captured.seen != 0)
    return fail("bad compressed checksum reached the codec");

  pivco_capture_destroy(&capture);

  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>transpose>pivco", width,
                            planes, "LINEAR:MAX_ERROR=2", GEOZL_DT_U16,
                            GEOZL_NODATA_NONE, 0, input, count,
                            sizeof(input[0]), error, sizeof(error)) != 0) {
    fprintf(stderr, "test_pivco_capture: %s\n", error);
    return 1;
  }
  if (geozl_2d_compress_graph_c(graph, input, count, frame, frameCapacity,
                                &frameSize, error, sizeof(error)) != 0) {
    fprintf(stderr, "test_pivco_capture: %s\n", error);
    return 1;
  }
  geozl_2d_graph_close_c(graph);
  if (pivco_capture_init(&capture, payload, frameCapacity + 64) != 0)
    return fail("could not recreate capture context");
  if (pivco_capture_frame(&capture, &captured, output, sizeof(output), frame,
                          frameSize) != PIVCO_CAPTURE_ERROR)
    return fail("wrapped frame was accepted");
  pivco_capture_destroy(&capture);

  if (check_transpose_codec("id>transpose>pivco", PIVCO_CAPTURE_CODEC) != 0 ||
      check_transpose_codec("delta_w>zigzag>transpose>pivco",
                            PIVCO_CAPTURE_ERROR) != 0)
    return 1;

  free(payload);
  free(inconsistentFrame);
  free(frame);
  puts("test_pivco_capture: ok");
  return 0;
}
