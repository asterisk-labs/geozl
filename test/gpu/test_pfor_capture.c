#include "gpu/pfor_capture.h"

#include "capture_frame_builder.h"

#include "common/endian.h"
#include "geozl/dtype.h"
#include "geozl/geozl.h"
#include "pfor/graph_pfor.h"

#include "openzl/zl_compress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
  fprintf(stderr, "test_pfor_capture: %s\n", message);
  return 1;
}

static int check_inconsistent_count(PforCapture *capture,
                                    CapturedPforFrame *captured, void *output,
                                    size_t outputSize, uint8_t *frame,
                                    size_t frameCapacity) {
  uint8_t header[PFOR_HEADER_SIZE] = {0};
  geozl_st_le64(header, 1024);
  header[8] = 1;
  const CaptureFramePlan plan = {
      .ctid = GEOZL_CTID_PFOR,
      .header = header,
      .headerSize = sizeof(header),
      .outputCount = 1,
      .outputBytes = {1, 0},
  };
  const size_t frameSize =
      capture_test_build_frame(frame, frameCapacity, outputSize, &plan);
  if (frameSize == 0)
    return fail("could not build an inconsistent frame");
  if (pfor_capture_frame(capture, captured, output, outputSize, frame,
                         frameSize) != PFOR_CAPTURE_ERROR ||
      captured->seen != -1 || captured->headerSize != 0 ||
      capture->payloadSize != 0)
    return fail("an inconsistent output size reached capture");
  return 0;
}

int main(void) {
  enum { width = 8, planes = 2, count = width * 8 * planes };
  uint16_t input[count];
  for (size_t i = 0; i < count; ++i)
    input[i] = (uint16_t)(i * 17 + i / width);

  char error[512];
  geozl_2d_graph *graph = NULL;
  if (geozl_2d_graph_open_c(&graph, "planar>zigzag>pfor", width, planes, NULL,
                            GEOZL_DT_U16, GEOZL_NODATA_NONE, 0, input, count,
                            sizeof(input[0]), error, sizeof(error)) != 0) {
    fprintf(stderr, "test_pfor_capture: %s\n", error);
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
    fprintf(stderr, "test_pfor_capture: %s\n", error);
    return 1;
  }
  geozl_2d_graph_close_c(graph);

  uint16_t output[count];
  memset(output, 0xA5, sizeof(output));
  PforCapture capture;
  CapturedPforFrame captured;
  if (pfor_capture_init(&capture, payload, frameCapacity + 64) != 0)
    return fail("could not create capture context");
  if (check_inconsistent_count(&capture, &captured, output, 64,
                               inconsistentFrame, frameCapacity) != 0)
    return 1;
  if (pfor_capture_frame(&capture, &captured, output, sizeof(output), frame,
                         frameSize) != 0)
    return fail("valid frame was not captured");
  for (size_t i = 0; i < sizeof(output); ++i) {
    if (((const uint8_t *)output)[i] != 0xA5)
      return fail("capture wrote the output buffer");
  }
  if (captured.headerSize != PLANAR_ZIGZAG_PFOR_HEADER_SIZE ||
      captured.kind != PFOR_CAPTURE_KIND_PLANAR_ZIGZAG ||
      captured.payloadSize == 0 || capture.payloadSize < captured.payloadSize)
    return fail("captured inputs are incomplete");

  geozl_2d_graph *idGraph = NULL;
  if (geozl_2d_graph_open_c(&idGraph, "id>pfor", width, planes, NULL,
                            GEOZL_DT_U16, GEOZL_NODATA_NONE, 0, input, count,
                            sizeof(input[0]), error, sizeof(error)) != 0)
    return fail("could not build id>pfor");
  if (geozl_2d_compress_graph_c(idGraph, input, count, frame, frameCapacity,
                                &frameSize, error, sizeof(error)) != 0) {
    geozl_2d_graph_close_c(idGraph);
    return fail("could not compress id>pfor");
  }
  geozl_2d_graph_close_c(idGraph);
  if (pfor_capture_frame(&capture, &captured, output, sizeof(output), frame,
                         frameSize) != PFOR_CAPTURE_CODEC ||
      captured.kind != PFOR_CAPTURE_KIND_ID ||
      captured.headerSize != PFOR_HEADER_SIZE)
    return fail("id>pfor was not captured");

  frame[frameSize - 2] ^= 1;
  if (pfor_capture_frame(&capture, &captured, output, sizeof(output), frame,
                         frameSize) == 0 ||
      captured.seen != 0)
    return fail("bad compressed checksum reached the codec");

  pfor_capture_destroy(&capture);
  free(payload);
  free(inconsistentFrame);
  free(frame);
  puts("test_pfor_capture: ok");
  return 0;
}
