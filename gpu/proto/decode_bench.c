// Times the GPU decoders on the path a reader calls, from OpenZL frames to
// samples in device memory. Every call captures the frames, plans on the host,
// uploads and runs the kernels. The first call sizes the decoder's staging and
// the last reuses it; both are checked against the source.

#include "bench.h"
#include "gpu.h"
#include "gpu/pfor_decode.h"
#include "gpu/pivco_decode.h"

#include "geozl/dtype.h"
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  const Shape *shapes;
  const void *const *frames;
  const size_t *frameSizes;
  size_t eltBytes;
  uint8_t *out;
} Frames;

static int decode_range(void *ctx, size_t begin, size_t end) {
  const Frames *fr = ctx;
  char err[256];
  for (size_t f = begin; f < end; ++f) {
    const size_t bytes = fr->shapes[f].nbElts * fr->eltBytes;
    size_t got = 0;
    if (geozl_2d_decompress_c(fr->frames[f], fr->frameSizes[f],
                              fr->out + fr->shapes[f].outOff * fr->eltBytes,
                              bytes, &got, 0, err, sizeof err) != 0 ||
        got != bytes)
      return 1;
  }
  return 0;
}

typedef struct {
  GeozlPforGpuDecoder *pfor;
  GeozlPivcoGpuDecoder *pivco;
} Decoder;

static GeozlGpuMemory *gpu_decode(const Decoder *d, const Frames *fr, size_t n,
                                  size_t *outputBytes, char *error,
                                  size_t errorCapacity) {
  if (d->pfor != NULL)
    return geozl_pfor_gpu_decode(d->pfor, fr->frames, fr->frameSizes, n, 0,
                                 outputBytes, error, errorCapacity);
  return geozl_pivco_gpu_decode(d->pivco, fr->frames, fr->frameSizes, n, 0,
                                outputBytes, error, errorCapacity);
}

static int compare_doubles(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  const char *codec = arg(argc, argv, "--codec", "pfor");
  const char *input = arg(argc, argv, "--input", NULL);
  const char *modulePath = arg(argc, argv, "--module", NULL);
  const size_t cells = (size_t)atol(arg(argc, argv, "--cells", "0"));
  const size_t bands = (size_t)atol(arg(argc, argv, "--bands", "4"));
  const size_t side = (size_t)atol(arg(argc, argv, "--side", "1056"));
  const size_t tile = (size_t)atol(arg(argc, argv, "--tile", "176"));
  const int wholeCells = strcmp(arg(argc, argv, "--mode", "tile"), "cell") == 0;
  const int reps = atoi(arg(argc, argv, "--reps", "20"));
  const int u8 = flag(argc, argv, "--u8");
  const int pivco = strcmp(codec, "pivco") == 0;
  if (!input || !modulePath || cells == 0 || tile == 0 || side % tile != 0 ||
      reps <= 0 || (!pivco && strcmp(codec, "pfor") != 0)) {
    fprintf(stderr,
            "usage: --codec pfor|pivco --input raw.u16|synthetic --cells N "
            "--module pzp2|pzv2.metal|.ptx [--bands 4 --side 1056 --tile 176 "
            "--mode tile|cell --u8 --reps 20]\n");
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
  const size_t W = u8 ? 1 : 2, outBytes = cells * bands * side * side * W;

  char err[512];
  geozl_2d_graph *g = NULL;
  const char *recipe =
      pivco ? "planar>zigzag>transpose>pivco" : "planar>zigzag>pfor";
  if (geozl_2d_graph_open_c(
          &g, recipe, (uint32_t)shapes[0].width, shapes[0].planes, NULL,
          u8 ? GEOZL_DT_U8 : GEOZL_DT_U16, GEOZL_NODATA_NONE, 0, orig,
          shapes[0].nbElts, W, err, sizeof err) != 0) {
    fprintf(stderr, "graph: %s\n", err);
    return 1;
  }
  size_t bound = 0;
  for (size_t f = 0; f < nFrames; ++f)
    bound += ZL_compressBound(shapes[f].nbElts * W);
  uint8_t *frames = malloc(bound);
  const void **framePointers = malloc(nFrames * sizeof(*framePointers));
  size_t *frameSizes = malloc(nFrames * sizeof(*frameSizes));
  if (frames == NULL || framePointers == NULL || frameSizes == NULL)
    return 1;
  size_t pos = 0;
  for (size_t f = 0; f < nFrames; ++f) {
    if (geozl_2d_compress_graph_c(g, orig + shapes[f].outOff * W,
                                  shapes[f].nbElts, frames + pos, bound - pos,
                                  &frameSizes[f], err, sizeof err) != 0) {
      fprintf(stderr, "frame %zu: %s\n", f, err);
      return 1;
    }
    framePointers[f] = frames + pos;
    pos += frameSizes[f];
  }
  geozl_2d_graph_close_c(g);
  const double rawMB = (double)outBytes / 1e6;
  printf("data      %s, %zu frames (%s, width %zu), %zu-byte samples, %.1f MB "
         "raw, %.1f MB in frames\n",
         recipe, nFrames, wholeCells ? "cell" : "tile", shapes[0].width, W,
         rawMB, (double)pos / 1e6);

  uint8_t *ref = page_alloc(outBytes, NULL);
  Frames fr = {shapes, framePointers, frameSizes, W, ref};
  const int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
  const double cpu1 =
      cpu_best(decode_range, &fr, nFrames, 1, reps < 5 ? reps : 5);
  if (cpu1 < 0 || memcmp(ref, orig, outBytes) != 0) {
    fprintf(stderr, "CPU reference does not round trip\n");
    return 1;
  }
  const double cpuN = cpu_best(decode_range, &fr, nFrames, cores, reps);
  printf("cpu       geozl frames, 1 thread   %8.2f ms  %7.1f GB/s\n",
         cpu1 * 1e3, rawMB / 1e3 / cpu1);
  printf("cpu       geozl frames, %2d threads  %8.2f ms  %7.1f GB/s\n", cores,
         cpuN * 1e3, rawMB / 1e3 / cpuN);

  size_t moduleSize;
  void *module = gpu_load_module(modulePath, &moduleSize);
  if (module == NULL) {
    fprintf(stderr, "cannot read %s\n", modulePath);
    return 1;
  }
  const char *deviceEnv = getenv("GPU");
  const int device = deviceEnv == NULL ? 0 : atoi(deviceEnv);
  Decoder decoder = {0};
  if (pivco)
    decoder.pivco = geozl_pivco_gpu_decoder_create(
        GPU_BACKEND, module, moduleSize, W, device, err, sizeof err);
  else
    decoder.pfor = geozl_pfor_gpu_decoder_create(
        GPU_BACKEND, module, moduleSize, W, device, err, sizeof err);
  free(module);
  if (decoder.pfor == NULL && decoder.pivco == NULL) {
    fprintf(stderr, "%s\n", err);
    return 1;
  }

  double *times = malloc((size_t)reps * sizeof(*times));
  uint8_t *got = malloc(outBytes);
  if (times == NULL || got == NULL)
    return 1;
  int mismatch = 0;
  for (int r = 0; r < reps; ++r) {
    size_t outputBytes = 0;
    const double t0 = now_seconds();
    GeozlGpuMemory *memory =
        gpu_decode(&decoder, &fr, nFrames, &outputBytes, err, sizeof err);
    times[r] = now_seconds() - t0;
    if (memory == NULL) {
      fprintf(stderr, "%s\n", err);
      return 1;
    }
    if (r == 0 || r == reps - 1)
      mismatch |= outputBytes != outBytes ||
                  geozl_gpu_memory_download(memory, got, outBytes, err,
                                            sizeof err) != 0 ||
                  memcmp(got, orig, outBytes) != 0;
    geozl_gpu_memory_destroy(memory);
  }
  const double first = times[0];
  qsort(times, (size_t)reps, sizeof(*times), compare_doubles);
  printf("gpu       %s decode call  first %8.2f ms  best %8.2f ms  median "
         "%8.2f ms  %7.1f GB/s\n",
         GPU_BACKEND == GEOZL_GPU_BACKEND_METAL ? "metal" : "cuda", first * 1e3,
         times[0] * 1e3, times[reps / 2] * 1e3, rawMB / 1e3 / times[0]);
  printf("check     %s\n", mismatch ? "MISMATCH against the source samples"
                                    : "bit-identical to the source samples");
  geozl_pfor_gpu_decoder_destroy(decoder.pfor);
  geozl_pivco_gpu_decoder_destroy(decoder.pivco);
  return mismatch;
}
