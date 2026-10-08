// Benchmark of the planar_zigzag_pfor GPU decoder: encodes frames with GeoZL's
// kernel, decodes them on the CPU as the reference, and hands the GPU the
// payload, one task per PFOR block and one per plane.

#include "bench.h"
#include "gpu.h"

#include "pfor/pfor_check.h"
#include "planar_zigzag_pfor/decode_planar_zigzag_pfor_kernel.h"
#include "planar_zigzag_pfor/encode_planar_zigzag_pfor_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Mirrors block_task in planar_zigzag_pfor.slang.
typedef struct {
  uint32_t payOff;
  uint32_t outBase;
  uint32_t count;
} BlockTask;

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
    bound += planar_zigzag_pfor_bound(shapes[f].nbElts, W);
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

  // A block's size is known only from its header, so the host walks every
  // frame's blocks once.
  const double tp = now_seconds();
  size_t nBlocks = 0;
  for (size_t f = 0; f < nFrames; ++f)
    nBlocks += (shapes[f].nbElts + GEOZL_PFOR_BLOCK - 1) / GEOZL_PFOR_BLOCK;
  // Kernels run 4 blocks per group; the padding blocks write nothing.
  BlockTask *blocks = calloc((nBlocks + 3) / 4 * 4, sizeof(BlockTask));
  size_t nb = 0, withExc = 0, exceptions = 0;
  for (size_t f = 0; f < nFrames; ++f) {
    const Shape *s = &shapes[f];
    const uint8_t *p = payload + payOff[f];
    size_t at = 0;
    for (size_t k = 0; k * GEOZL_PFOR_BLOCK < s->nbElts; ++k) {
      const size_t left = s->nbElts - k * GEOZL_PFOR_BLOCK;
      blocks[nb++] = (BlockTask){
          (uint32_t)(payOff[f] + at),
          (uint32_t)(s->outOff + k * GEOZL_PFOR_BLOCK),
          (uint32_t)(left < GEOZL_PFOR_BLOCK ? left : GEOZL_PFOR_BLOCK)};
      PforBlockInfo block;
      if (pfor_block_info(p + at, paySize[f] - at, W, &block) != 0) {
        fprintf(stderr, "invalid block %zu in frame %zu at byte %zu\n", k, f,
                at);
        return 1;
      }
      if (block.exceptions != 0) {
        withExc++;
        exceptions += block.exceptions;
      }
      at += block.bytes;
    }
    if (at != paySize[f]) {
      fprintf(stderr, "block walk of frame %zu ended at byte %zu of %zu\n", f,
              at, paySize[f]);
      return 1;
    }
  }
  nBlocks = (nb + 3) / 4 * 4;
  const Planes planes = plane_tasks(shapes, nFrames);
  const double planS = now_seconds() - tp;
  printf("plan      %zu block tasks, %zu plane tasks, host walk %.2f ms\n",
         nBlocks, planes.n, planS * 1e3);
  printf("blocks    %.1f%% with exceptions, %.2f%% of values are exceptions\n",
         100.0 * (double)withExc / (double)nb,
         100.0 * (double)exceptions / (double)outElts);

  size_t outAlloc;
  uint8_t *out = page_alloc(outElts * W, &outAlloc);
  // In declaration order in planar_zigzag_pfor.slang.
  GpuBuffer buffers[] = {
      {payload, page_round(pos + 64), 1, 0, -1},
      {blocks, nBlocks * sizeof(BlockTask), sizeof(BlockTask), 0, -1},
      {planes.tasks, planes.n * sizeof(PlaneTask), sizeof(PlaneTask), 0, -1},
      {out, outAlloc, W, 1, -1},
      {NULL, outAlloc, 1, 1, 3}, // the output again, as bytes
  };
  GpuDispatch dispatches[4] = {
      {"unpack_blocks", (unsigned)(nBlocks / 4), 1, 128}};
  const int nd = 1 + scan_dispatches(&planes, dispatches + 1);
  GpuTimes gt = {0};
  if (gpu_run(module, buffers, 5, dispatches, nd, reps, &gt) != 0) {
    fprintf(stderr, "GPU run failed\n");
    return 1;
  }
  return report_gpu(&gt, dispatches, nd, rawMB, planS, out, orig, outElts * W);
}
