// CPU/GPU benchmark for planar_zigzag_pfor.

#include "bench.h"
#include "gpu.h"
#include "gpu/pfor_plan.h"

#include "planar_zigzag_pfor/decode_planar_zigzag_pfor_kernel.h"
#include "planar_zigzag_pfor/encode_planar_zigzag_pfor_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  const Shape *shapes;
  const size_t *payOff, *paySize;
  const uint8_t *payload;
  size_t eltBytes;
  uint8_t *out;
} Frames;

static int decode_range(void *ctx, size_t begin, size_t end) {
  const Frames *fr = ctx;
  for (size_t f = begin; f < end; ++f) {
    const Shape *s = &fr->shapes[f];
    if (planar_zigzag_pfor_decode(fr->out + s->outOff * fr->eltBytes, s->width,
                                  s->nbElts, fr->eltBytes, s->planes,
                                  fr->payload + fr->payOff[f],
                                  fr->paySize[f]) != 0)
      return 1;
  }
  return 0;
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
        "usage: --input raw.u16|synthetic --cells N --module pzp2.metal|.ptx "
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

  size_t bound = 0;
  for (size_t f = 0; f < nFrames; ++f)
    bound += planar_zigzag_pfor_bound(shapes[f].nbElts, W, shapes[f].width,
                                      shapes[f].planes);
  // Zeroed padding past the end for the kernels' unaligned loads.
  size_t payloadAlloc;
  uint8_t *payload = page_alloc(bound + 64, &payloadAlloc);
  size_t *payOff = malloc(nFrames * sizeof(size_t)),
         *paySize = malloc(nFrames * sizeof(size_t));
  const double te = now_seconds();
  size_t pos = 0;
  for (size_t f = 0; f < nFrames; ++f) {
    const Shape *s = &shapes[f];
    if (planar_zigzag_pfor_encode(payload + pos, bound - pos, &paySize[f],
                                  orig + s->outOff * W, s->width, s->nbElts, W,
                                  s->planes) != 0) {
      fprintf(stderr, "encode failed on frame %zu\n", f);
      return 1;
    }
    payOff[f] = pos;
    pos += paySize[f];
  }
  const double encodeS = now_seconds() - te;
  const double rawMB = (double)(outElts * W) / 1e6;
  printf("data      %zu frames (%s, width %zu), %zu-byte samples, %.1f MB raw, "
         "%.1f MB payload, ratio %.3f, encode %.2f s\n",
         nFrames, wholeCells ? "cell" : "tile", shapes[0].width, W, rawMB,
         (double)pos / 1e6, (double)(outElts * W) / (double)pos, encodeS);

  uint8_t *ref = page_alloc(outElts * W, NULL);
  Frames fr = {shapes, payOff, paySize, payload, W, ref};
  const int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
  const double cpu1 =
      cpu_best(decode_range, &fr, nFrames, 1, reps < 5 ? reps : 5);
  if (cpu1 < 0 || memcmp(ref, orig, outElts * W) != 0) {
    fprintf(stderr, "CPU reference does not round trip\n");
    return 1;
  }
  const double cpuN = cpu_best(decode_range, &fr, nFrames, cores, reps);
  printf("cpu       geozl, 1 thread   %8.2f ms  %7.1f GB/s\n", cpu1 * 1e3,
         rawMB / 1e3 / cpu1);
  printf("cpu       geozl, %2d threads  %8.2f ms  %7.1f GB/s\n", cores,
         cpuN * 1e3, rawMB / 1e3 / cpuN);

  const double tp = now_seconds();
  PforGpuFrame *gpuFrames = calloc(nFrames, sizeof(*gpuFrames));
  if (gpuFrames == NULL)
    return 1;
  for (size_t f = 0; f < nFrames; ++f) {
    const Shape *s = &shapes[f];
    gpuFrames[f] = (PforGpuFrame){{s->outOff, s->nbElts, s->width, s->planes},
                                  payOff[f],
                                  paySize[f],
                                  PFOR_GPU_PLANAR_ZIGZAG};
  }
  PforGpuBandPlan plan;
  if (pfor_gpu_band_plan_build(&plan, gpuFrames, nFrames, NULL, NULL) != 0) {
    if (plan.errorFrame == SIZE_MAX)
      fprintf(stderr, "%s\n", plan.error);
    else
      fprintf(stderr, "%s in frame %zu\n", plan.error, plan.errorFrame);
    free(gpuFrames);
    return 1;
  }
  free(gpuFrames);
  const double planS = now_seconds() - tp;
  size_t withExceptions = 0, exceptions = 0;
  for (size_t f = 0; f < plan.frameCount; ++f) {
    const GpuPforBandFrame *frame = &plan.frames[f];
    for (size_t k = 0; k < frame->blockCount; ++k) {
      const unsigned nexc = payload[frame->payloadOffset + 3 * k + 1];
      withExceptions += nexc != 0;
      exceptions += nexc;
    }
  }
  printf("plan      %zu strip tasks, %zu blocks, host layout %.2f ms\n",
         plan.stripCount, plan.blockCount, planS * 1e3);
  printf("blocks    %.1f%% with exceptions, %.2f%% of values are exceptions\n",
         100.0 * (double)withExceptions / (double)plan.blockCount,
         100.0 * (double)exceptions / (double)outElts);

  size_t outAlloc;
  uint8_t *out = page_alloc(outElts * W, &outAlloc);
  uint32_t dummy[4] = {0};
  uint32_t status[2] = {UINT32_MAX, UINT32_MAX};
  // In declaration order in kernels/programs/pfor.slang.
  GpuBuffer buffers[] = {
      {payload, page_round(pos + 64), 1, 0, -1},
      {dummy, sizeof(dummy), sizeof(GpuPforBlock), 0, -1},
      {out, outAlloc, W, 1, -1},
      {NULL, outAlloc, 1, 1, 2}, // the output again, as bytes
      {plan.frames, plan.frameCount * sizeof(GpuPforBandFrame),
       sizeof(GpuPforBandFrame), 0, -1},
      {plan.strips, plan.stripCount * sizeof(GpuPforStrip),
       sizeof(GpuPforStrip), 0, -1},
      {NULL, plan.blockCount * sizeof(GpuPforBlockRef), sizeof(GpuPforBlockRef),
       0, -1},
      {status, sizeof(status), sizeof(uint32_t), 0, -1},
      {plan.chunks, plan.chunkCount * sizeof(GpuPforBlockChunk),
       sizeof(GpuPforBlockChunk), 0, -1},
      {NULL, plan.chunkCount * sizeof(uint32_t), sizeof(uint32_t), 0, -1},
      {dummy, 1, 1, 0, -1},
      {dummy, 12, 12, 0, -1},
  };
  const GpuDispatch dispatches[] = {
      {"size_chunks", (unsigned)(plan.chunkCount / 4), 1, 128},
      {"plan_blocks", (unsigned)(plan.chunkCount / 4), 1, 128},
      {"decode_strips", (unsigned)(plan.stripCount / 4), 1, 128},
  };
  GpuTimes gt = {0};
  if (gpu_run(module, buffers, 12, dispatches, 3, reps, &gt) != 0) {
    fprintf(stderr, "GPU run failed\n");
    return 1;
  }
  return report_gpu(&gt, dispatches, 3, rawMB, planS, out, orig, outElts * W);
}
