// Benchmark of the planar_zigzag_pivco GPU decoder on real frames. GeoZL
// compresses every frame with the recipe planar>zigzag>pivco and OpenZL decodes
// them on the CPU as the reference. A decoder registered in place of the
// codec's records the header and the two streams OpenZL hands it, which is all
// the GPU decodes.

#include "bench.h"
#include "gpu.h"
#include "pivco_plan.h"

#include "geozl/ctids.h"
#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "planar_zigzag_pivco/decode_planar_zigzag_pivco_binding.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_input.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static struct {
  CapturedFrame *frame; // the frame being decoded
  uint8_t *payload;
  size_t pos, capacity;
} spy;

static ZL_Report spy_decoder(ZL_Decoder *d, const ZL_Input *ins[]) {
  CapturedFrame *c = spy.frame;
  const ZL_RBuffer h = ZL_Decoder_getCodecHeader(d);
  const size_t w = ZL_Input_numElts(ins[0]), b = ZL_Input_numElts(ins[1]);
  const size_t padded = b <= SIZE_MAX - 15 ? (b + 15) / 16 * 16 : SIZE_MAX;
  if (c->seen || h.size > sizeof c->header || w > sizeof c->weights ||
      spy.pos > spy.capacity || padded > spy.capacity - spy.pos) {
    c->seen = -1;
  } else {
    c->seen = 1;
    memcpy(c->header, h.start, h.size);
    c->headerSize = h.size;
    memcpy(c->weights, ZL_Input_ptr(ins[0]), w);
    c->weightsSize = w;
    memcpy(spy.payload + spy.pos, ZL_Input_ptr(ins[1]), b);
    c->bitsOffset = spy.pos;
    c->bitsSize = b;
    spy.pos += padded;
  }
  return DI_geozl_planar_zigzag_pivco(d, ins);
}

// Every GeoZL decoder, or only the spy in place of this codec's. Neither checks
// checksums, as the GPU does not.
static ZL_DCtx *open_dctx(int withSpy) {
  ZL_DCtx *d = ZL_DCtx_create();
  if (d == NULL)
    return NULL;
  ZL_Report r;
  if (withSpy) {
    ZL_TypedDecoderDesc desc =
        DI_PLANAR_ZIGZAG_PIVCO(GEOZL_CTID_PLANAR_ZIGZAG_PIVCO);
    desc.transform_f = spy_decoder;
    r = ZL_DCtx_registerTypedDecoder(d, &desc);
  } else {
    r = geozl_register_decoders(d);
  }
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
  ZL_DCtx *d = open_dctx(0);
  if (d == NULL)
    return 1;
  int bad = 0;
  for (size_t f = begin; f < end && !bad; ++f)
    bad = decode_one(d, ctx, f);
  ZL_DCtx_free(d);
  return bad;
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
  if (geozl_2d_graph_open_c(&g, "planar>zigzag>pivco",
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

  // Record what the codec's decoder receives. The bitstreams are at most the
  // frames, and each starts on 16 bytes.
  uint8_t *ref = page_alloc(outElts * W, NULL);
  Frames fr = {shapes, frames, frameOff, frameSize, W, ref};
  CapturedFrame *cap = calloc(nFrames, sizeof(*cap));
  spy.capacity = pos + 16 * nFrames;
  spy.payload = page_alloc(spy.capacity + 64, NULL);
  ZL_DCtx *sd = open_dctx(1);
  for (size_t f = 0; f < nFrames; ++f) {
    spy.frame = &cap[f];
    if (sd == NULL || decode_one(sd, &fr, f) != 0 || cap[f].seen != 1) {
      fprintf(stderr,
              "frame %zu does not decode through planar_zigzag_pivco alone\n",
              f);
      return 1;
    }
  }
  ZL_DCtx_free(sd);

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

  // Translate the validated codec inputs into the immutable task lists that
  // the GPU kernels consume.
  const double tp = now_seconds();
  PivcoGpuPlan plan;
  if (pivco_gpu_plan_build(&plan, cap, shapes, nFrames, spy.payload, spy.pos,
                           W) != 0)
    return 1;
  const Planes planes = plane_tasks(shapes, nFrames);
  const double planS = now_seconds() - tp;
  printf("lanes     %zu PivCo (%zu blocks, %zu bitmaps, %.1f MB rank "
         "directory), %zu PFOR "
         "(%zu blocks)\n",
         plan.pivcoLaneCount, plan.pivcoBlockCount, plan.internalNodeCount,
         (double)plan.rankWordCount * 8 / 1e6, plan.pfor_laneCount,
         plan.laneBlockCount);
  printf("plan      %zu frames, %zu plane tasks, host plan %.2f ms\n", nFrames,
         planes.n, planS * 1e3);

  size_t outAlloc;
  uint8_t *out = page_alloc(outElts * W, &outAlloc);
  // In declaration order in planar_zigzag_pivco.slang.
  GpuBuffer buffers[] = {
      {spy.payload, page_round(spy.pos + 64), 1, 0, -1},
      {plan.symbols, (plan.symbolCount ? plan.symbolCount : 1), 1, 0, -1},
      {plan.trees, (plan.treeCount ? plan.treeCount : 1) * sizeof(GpuTreeNode),
       sizeof(GpuTreeNode), 0, -1},
      {plan.nodes, (plan.nodeCount ? plan.nodeCount : 1) * sizeof(GpuNodeBits),
       sizeof(GpuNodeBits), 0, -1},
      {plan.internalNodes,
       (plan.internalNodeCount ? plan.internalNodeCount : 1) * sizeof(uint32_t),
       sizeof(uint32_t), 0, -1},
      {plan.pivcoBlocks,
       (plan.pivcoBlockCount ? plan.pivcoBlockCount : 1) *
           sizeof(GpuPivcoBlock),
       sizeof(GpuPivcoBlock), 0, -1},
      {NULL,
       (plan.rankWordCount ? plan.rankWordCount : 1) * 2 * sizeof(uint32_t),
       2 * sizeof(uint32_t), 0, -1},
      {NULL, plan.laneBytes + 256, 1, 0, -1},
      {plan.laneBlocks,
       (plan.laneBlockCount ? plan.laneBlockCount : 1) * sizeof(GpuLaneBlock),
       sizeof(GpuLaneBlock), 0, -1},
      {plan.joins, nFrames * sizeof(GpuJoinTask), sizeof(GpuJoinTask), 0, -1},
      {planes.tasks, planes.n * sizeof(PlaneTask), sizeof(PlaneTask), 0, -1},
      {out, outAlloc, W, 1, -1},
      {NULL, outAlloc, 1, 1, 11}, // the output again, as bytes
  };
  const unsigned perGroup = 128 * 8; // threads per group times bytes per thread
  GpuDispatch d[GPU_MAX_DISPATCHES];
  int nd = 0;
  if (plan.internalNodeCount)
    d[nd++] = (GpuDispatch){"build_ranks",
                            (unsigned)(plan.internalNodeCount / 4), 1, 128};
  if (plan.pivcoBlockCount)
    d[nd++] = (GpuDispatch){
        "decode_symbols", (unsigned)plan.pivcoBlockCount,
        (unsigned)((plan.maxPivcoBlockLength + perGroup - 1) / perGroup), 128};
  if (plan.laneBlockCount)
    d[nd++] = (GpuDispatch){"unpack_lane_blocks",
                            (unsigned)(plan.laneBlockCount / 4), 1, 128};
  d[nd++] = (GpuDispatch){
      "join_lanes", (unsigned)nFrames,
      (unsigned)((plan.maxFrameLength + perGroup - 1) / perGroup), 128};
  nd += scan_dispatches(&planes, d + nd);
  GpuTimes gt = {0};
  if (gpu_run(module, buffers, 13, d, nd, reps, &gt) != 0) {
    fprintf(stderr, "GPU run failed\n");
    return 1;
  }
  return report_gpu(&gt, d, nd, rawMB, planS, out, orig, outElts * W);
}
