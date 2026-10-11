// Benchmark of the planar_zigzag_transpose_pivco GPU decoder on real frames.
// GeoZL compresses every frame with planar>zigzag>transpose>pivco and OpenZL
// decodes them on the CPU as the reference. A decoder registered in place of
// the codec's records the header and the two streams OpenZL hands it, which is
// all the GPU decodes.

#include "bench.h"
#include "gpu.h"
#include "gpu/pivco_capture.h"
#include "gpu/pivco_decode.h"
#include "gpu/pivco_plan.h"

#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_decompress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ZL_DCtx *open_dctx(void) {
  ZL_DCtx *d = ZL_DCtx_create();
  if (d == NULL)
    return NULL;
  ZL_Report r = geozl_register_decoders(d);
  if (!ZL_isError(r))
    r = ZL_DCtx_setParameter(d, ZL_DParam_stickyParameters, 1);
  if (!ZL_isError(r))
    r = ZL_DCtx_setParameter(d, ZL_DParam_checkCompressedChecksum,
                             ZL_TernaryParam_disable);
  if (!ZL_isError(r))
    r = ZL_DCtx_setParameter(d, ZL_DParam_checkContentChecksum,
                             ZL_TernaryParam_disable);
  if (ZL_isError(r)) {
    ZL_DCtx_free(d);
    return NULL;
  }
  return d;
}

typedef struct {
  const Shape *shapes;
  const uint8_t *frames;
  const size_t *frameOff, *frameSize;
  size_t eltBytes;
  uint8_t *out;
} Frames;

static int decode_one(ZL_DCtx *d, const Frames *fr, size_t f) {
  const Shape *s = &fr->shapes[f];
  const size_t bytes = s->nbElts * fr->eltBytes;
  ZL_OutputInfo info;
  const ZL_Report r = ZL_DCtx_decompressTyped(
      d, &info, fr->out + s->outOff * fr->eltBytes, bytes,
      fr->frames + fr->frameOff[f], fr->frameSize[f]);
  return ZL_isError(r) || info.decompressedByteSize != bytes;
}

static int decode_range(void *ctx, size_t begin, size_t end) {
  ZL_DCtx *d = open_dctx();
  if (d == NULL)
    return 1;
  int bad = 0;
  for (size_t f = begin; f < end && !bad; ++f)
    bad = decode_one(d, ctx, f);
  ZL_DCtx_free(d);
  return bad;
}

static int check_owned_gpu_decode(const char *modulePath, size_t elementBytes,
                                  const uint8_t *frames,
                                  const size_t frameOff[],
                                  const size_t frameSize[], size_t frameCount,
                                  const uint8_t *want, size_t wantBytes) {
  size_t moduleSize;
  void *module = gpu_load_module(modulePath, &moduleSize);
  const void **framePointers = calloc(frameCount, sizeof(*framePointers));
  uint8_t *got = malloc(wantBytes);
  char error[512];
  if (module == NULL || framePointers == NULL || got == NULL) {
    free(module);
    free(framePointers);
    free(got);
    return 1;
  }
  for (size_t i = 0; i < frameCount; ++i)
    framePointers[i] = frames + frameOff[i];

  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  GeozlPivcoGpuDecoder *decoder = geozl_pivco_gpu_decoder_create(
      module, moduleSize, elementBytes, device, error, sizeof(error));
  free(module);
  if (decoder == NULL) {
    fprintf(stderr, "%s\n", error);
    free(framePointers);
    free(got);
    return 1;
  }
  size_t outputBytes;
  GeozlGpuMemory *memory =
      geozl_pivco_gpu_decode(decoder, framePointers, frameSize, frameCount,
                             wantBytes, &outputBytes, error, sizeof(error));
  GeozlGpuMemory *overLimit =
      geozl_pivco_gpu_decode(decoder, framePointers, frameSize, frameCount,
                             wantBytes - 1, NULL, error, sizeof(error));
  geozl_pivco_gpu_decoder_destroy(decoder);
  free(framePointers);
  if (memory == NULL || overLimit != NULL) {
    geozl_gpu_memory_destroy(overLimit);
    fprintf(stderr, "%s\n",
            overLimit != NULL ? "output limit was ignored" : error);
    geozl_gpu_memory_destroy(memory);
    free(got);
    return 1;
  }
  error[0] = '\0';
  const int failed =
      outputBytes != wantBytes ||
      geozl_gpu_memory_download(memory, got, wantBytes, error, sizeof(error)) ||
      memcmp(got, want, wantBytes) != 0;
  if (failed)
    fprintf(stderr, "%s%s\n", error[0] ? error : "GPU output mismatch",
            outputBytes != wantBytes ? " (wrong output size)" : "");
  geozl_gpu_memory_destroy(memory);
  free(got);
  return failed;
}

int main(int argc, char **argv) {
  const char *input = arg(argc, argv, "--input", NULL);
  const char *module = arg(argc, argv, "--module", NULL);
  const size_t cells = (size_t)atol(arg(argc, argv, "--cells", "0"));
  const size_t bands = (size_t)atol(arg(argc, argv, "--bands", "4"));
  const size_t side = (size_t)atol(arg(argc, argv, "--side", "1056"));
  const size_t tile = (size_t)atol(arg(argc, argv, "--tile", "176"));
  const int wholeCells = strcmp(arg(argc, argv, "--mode", "tile"), "cell") == 0;
  const int reps = atoi(arg(argc, argv, "--reps", "20"));
  const int u8 = flag(argc, argv, "--u8");
  if (!input || !module || cells == 0 || tile == 0 || side % tile != 0) {
    fprintf(
        stderr,
        "usage: --input raw.u16|synthetic --cells N --module pzv2.metal|.ptx "
        "[--bands 4 --side 1056 --tile 176 --mode tile|cell --u8 --reps 20]\n");
    return 2;
  }
  uint16_t *raw = load_cells(input, cells, bands, side);
  if (!raw)
    return 1;
  Shape *shapes;
  size_t nFrames;
  uint8_t *orig = cut_frames(raw, cells, bands, side, tile, wholeCells, u8,
                             &shapes, &nFrames);
  free(raw);
  const size_t W = u8 ? 1 : 2, outElts = cells * bands * side * side;

  // Every frame has the same geometry, so one graph serves them all.
  char err[512];
  geozl_2d_graph *g = NULL;
  if (geozl_2d_graph_open_c(&g, "planar>zigzag>transpose>pivco",
                            (uint32_t)shapes[0].width, shapes[0].planes, NULL,
                            u8 ? GEOZL_DT_U8 : GEOZL_DT_U16, GEOZL_NODATA_NONE,
                            0, orig, shapes[0].nbElts, W, err,
                            sizeof err) != 0) {
    fprintf(stderr, "graph: %s\n", err);
    return 1;
  }
  size_t bound = 0;
  for (size_t f = 0; f < nFrames; ++f)
    bound += ZL_compressBound(shapes[f].nbElts * W);
  uint8_t *frames = malloc(bound);
  size_t *frameOff = malloc(nFrames * sizeof(size_t)),
         *frameSize = malloc(nFrames * sizeof(size_t));
  const double te = now_seconds();
  size_t pos = 0;
  for (size_t f = 0; f < nFrames; ++f) {
    const Shape *s = &shapes[f];
    if (geozl_2d_compress_graph_c(g, orig + s->outOff * W, s->nbElts,
                                  frames + pos, bound - pos, &frameSize[f], err,
                                  sizeof err) != 0) {
      fprintf(stderr, "frame %zu: %s\n", f, err);
      return 1;
    }
    frameOff[f] = pos;
    pos += frameSize[f];
  }
  const double encodeS = now_seconds() - te;
  geozl_2d_graph_close_c(g);
  const double rawMB = (double)(outElts * W) / 1e6;
  printf("data      %zu frames (%s, width %zu), %zu-byte samples, %.1f MB raw, "
         "%.1f MB in frames, ratio %.3f, encode %.2f s\n",
         nFrames, wholeCells ? "cell" : "tile", shapes[0].width, W, rawMB,
         (double)pos / 1e6, (double)(outElts * W) / (double)pos, encodeS);

  // Record only the final codec inputs needed by the GPU.
  uint8_t *ref = page_alloc(outElts * W, NULL);
  Frames fr = {shapes, frames, frameOff, frameSize, W, ref};
  CapturedFrame *cap = calloc(nFrames, sizeof(*cap));
  const size_t payloadCapacity = pos + 64;
  uint8_t *payload = page_alloc(payloadCapacity, NULL);
  PivcoCapture capture;
  if (pivco_capture_init(&capture, payload, payloadCapacity) != 0)
    return 1;
  for (size_t f = 0; f < nFrames; ++f) {
    const Shape *s = &shapes[f];
    if (pivco_capture_frame(&capture, &cap[f], ref + s->outOff * W,
                            s->nbElts * W, frames + frameOff[f],
                            frameSize[f]) != 0) {
      fprintf(stderr,
              "frame %zu does not end in planar_zigzag_transpose_pivco\n", f);
      return 1;
    }
  }
  const size_t payloadSize = capture.payloadSize;
  pivco_capture_destroy(&capture);

  memset(ref, 0, outElts * W);
  const int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
  const double cpu1 =
      cpu_best(decode_range, &fr, nFrames, 1, reps < 5 ? reps : 5);
  if (cpu1 < 0 || memcmp(ref, orig, outElts * W) != 0) {
    fprintf(stderr, "CPU reference does not round trip\n");
    return 1;
  }
  const double cpuN = cpu_best(decode_range, &fr, nFrames, cores, reps);
  printf("cpu       geozl frames, 1 thread   %8.2f ms  %7.1f GB/s\n",
         cpu1 * 1e3, rawMB / 1e3 / cpu1);
  printf("cpu       geozl frames, %2d threads  %8.2f ms  %7.1f GB/s\n", cores,
         cpuN * 1e3, rawMB / 1e3 / cpuN);

  if (check_owned_gpu_decode(module, W, frames, frameOff, frameSize, nFrames,
                             orig, outElts * W) != 0)
    return 1;

  // Translate the validated codec inputs into the immutable task lists that
  // the GPU kernels consume.
  const double tp = now_seconds();
  GeozlGpuFrameShape *gpuFrames = calloc(nFrames, sizeof(*gpuFrames));
  if (gpuFrames == NULL)
    return 1;
  for (size_t f = 0; f < nFrames; ++f) {
    gpuFrames[f] = (GeozlGpuFrameShape){shapes[f].outOff, shapes[f].nbElts,
                                        shapes[f].width, shapes[f].planes};
  }
  PivcoGpuPlan plan;
  if (pivco_gpu_plan_build_parallel(&plan, cap, gpuFrames, nFrames, payload,
                                    payloadSize, W, 4) != 0) {
    if (plan.errorFrame == SIZE_MAX)
      fprintf(stderr, "%s\n", plan.error);
    else if (plan.errorLane == SIZE_MAX)
      fprintf(stderr, "%s in frame %zu\n", plan.error, plan.errorFrame);
    else
      fprintf(stderr, "%s in lane %zu of frame %zu\n", plan.error,
              plan.errorLane, plan.errorFrame);
    free(gpuFrames);
    return 1;
  }
  free(gpuFrames);
  const Planes planes = plane_tasks(shapes, nFrames);
  const double planS = now_seconds() - tp;
  const size_t planBytes = plan.symbolCount +
                           plan.treeCount * sizeof(*plan.trees) +
                           plan.nodeCount * sizeof(*plan.nodes) +
                           plan.pivcoBlockCount * sizeof(*plan.pivcoBlocks) +
                           plan.laneBlockCount * sizeof(*plan.laneBlocks) +
                           plan.joinCount * sizeof(*plan.joins);
  printf("lanes     %zu PivCo (%zu blocks, %zu bitmaps), %zu PFOR "
         "(%zu blocks)\n",
         plan.pivcoLaneCount, plan.pivcoBlockCount, plan.bitmapCount,
         plan.pforLaneCount, plan.laneBlockCount);
  printf("plan      %zu frames, %zu plane tasks, %.1f MB lists, host %.2f ms\n",
         nFrames, planes.n, (double)planBytes / 1e6, planS * 1e3);

  size_t outAlloc;
  const size_t laneBytes = plan.laneBytes + 256;
  const size_t outputCapacity =
      outElts * W > laneBytes ? outElts * W : laneBytes;
  uint8_t *out = page_alloc(outputCapacity, &outAlloc);
  uint32_t dummy[3] = {0};
  // In declaration order in kernels/programs/pivco.slang.
  GpuBuffer buffers[] = {
      {payload, page_round(payloadCapacity), 1, 0, -1},
      {plan.symbols, (plan.symbolCount > 4 ? plan.symbolCount : 4), 1, 0, -1},
      {plan.trees, (plan.treeCount ? plan.treeCount : 1) * sizeof(GpuTreeNode),
       sizeof(GpuTreeNode), 0, -1},
      {plan.nodes, (plan.nodeCount ? plan.nodeCount : 1) * sizeof(GpuNodeBits),
       sizeof(GpuNodeBits), 0, -1},
      {plan.pivcoBlocks,
       (plan.pivcoBlockCount ? plan.pivcoBlockCount : 1) *
           sizeof(GpuPivcoBlock),
       sizeof(GpuPivcoBlock), 0, -1},
      {NULL, laneBytes, 1, 0, -1},
      {NULL, laneBytes, 1, 0, 5},
      {plan.laneBlocks,
       (plan.laneBlockCount ? plan.laneBlockCount : 1) * sizeof(GpuLaneBlock),
       sizeof(GpuLaneBlock), 0, -1},
      {plan.joins, plan.joinCount * sizeof(GpuJoinTask), sizeof(GpuJoinTask), 0,
       -1},
      {planes.tasks, planes.n * sizeof(PlaneTask), sizeof(PlaneTask), 0, -1},
      {out, outAlloc, W, 1, -1},
      {NULL, outAlloc, 1, 1, 10}, // the output again, as bytes
      {NULL, outAlloc, 1, 0, 10}, // PivCo scratch before join_lanes
      {dummy, 1, 1, 0, -1},
      {dummy, sizeof(dummy), sizeof(dummy), 0, -1},
  };
  GpuDispatch d[GPU_MAX_DISPATCHES];
  int nd = 0;
  if (plan.pivcoBlockCount)
    d[nd++] =
        (GpuDispatch){"decode_symbols", (unsigned)plan.pivcoBlockCount, 1, 128};
  if (plan.laneBlockCount)
    d[nd++] = (GpuDispatch){"unpack_lane_blocks",
                            (unsigned)(plan.laneBlockCount / 4), 1, 128};
  d[nd++] = (GpuDispatch){"join_lanes", (unsigned)plan.joinCount, 1, 128};
  nd += scan_dispatches(&planes, d + nd);
  GpuTimes gt = {0};
  if (gpu_run(module, buffers, 15, d, nd, reps, &gt) != 0) {
    fprintf(stderr, "GPU run failed\n");
    return 1;
  }
  return report_gpu(&gt, d, nd, rawMB, planS, out, orig, outElts * W);
}
