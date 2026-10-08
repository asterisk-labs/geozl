#include "bench.h"

#include "pfor/pfor_check.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(PlaneTask) == 12, "PlaneTask must match Slang");

double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void *page_alloc(size_t bytes, size_t *allocated) {
  const size_t page = (size_t)sysconf(_SC_PAGESIZE);
  const size_t size = (bytes + page - 1) / page * page;
  void *p = NULL;
  if (posix_memalign(&p, page, size) != 0)
    return NULL;
  memset(p, 0, size);
  if (allocated)
    *allocated = size;
  return p;
}

size_t page_round(size_t bytes) {
  const size_t page = (size_t)sysconf(_SC_PAGESIZE);
  return (bytes + page - 1) / page * page;
}

// Sentinel-2-like reflectance: smooth terrain, field parcels with sharp edges,
// sensor noise and rare bright outliers, deterministic for every machine.
static uint32_t lcg(uint64_t *s) {
  *s = *s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(*s >> 33);
}

void synthesize(uint16_t *raw, size_t cells, size_t bands, size_t side) {
  uint64_t seed = 2026;
  for (size_t c = 0; c < cells; ++c)
    for (size_t b = 0; b < bands; ++b) {
      const double phase = (double)(lcg(&seed) % 1000) / 100.0;
      for (size_t y = 0; y < side; ++y)
        for (size_t x = 0; x < side; ++x) {
          const double terrain = 1500.0 + 700.0 *
                                              sin((double)x / 61.0 + phase) *
                                              cos((double)y / 47.0 - phase);
          const unsigned parcel =
              (unsigned)((x / 64) * 7 + (y / 48) * 13 + b) % 9u;
          int v =
              (int)terrain + (int)parcel * 180 + (int)(lcg(&seed) % 81u) - 40;
          if (lcg(&seed) % 1000u == 0)
            v += 6000 + (int)(lcg(&seed) % 4000u);
          raw[((c * bands + b) * side + y) * side + x] =
              (uint16_t)(v < 0       ? 0
                         : v > 65535 ? 65535
                                     : v);
        }
    }
}

uint16_t *load_cells(const char *path, size_t cells, size_t bands,
                     size_t side) {
  const size_t n = cells * bands * side * side;
  uint16_t *raw = malloc(n * 2);
  if (strcmp(path, "synthetic") == 0) {
    synthesize(raw, cells, bands, side);
    return raw;
  }
  FILE *fp = fopen(path, "rb");
  if (!fp || fread(raw, 2, n, fp) != n) {
    fprintf(stderr, "cannot read %zu samples from %s\n", n, path);
    free(raw);
    if (fp)
      fclose(fp);
    return NULL;
  }
  fclose(fp);
  return raw;
}

const char *arg(int argc, char **argv, const char *name, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], name) == 0)
      return argv[i + 1];
  return def;
}

int flag(int argc, char **argv, const char *name) {
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], name) == 0)
      return 1;
  return 0;
}

int pfor_block_info(const uint8_t *block, size_t available, size_t elementBytes,
                    PforBlockInfo *info) {
  if (block == NULL || info == NULL || !geozl_pfor_width_ok(elementBytes) ||
      available < 2)
    return 1;

  const unsigned bits = block[0];
  const unsigned exceptions = block[1];
  if (bits > elementBytes * 8)
    return 1;

  size_t exceptionBytes = 0;
  if (exceptions != 0) {
    if (available < 3)
      return 1;
    const unsigned descriptor = block[2];
    const unsigned exceptionBits = descriptor & 0x7Fu;
    const unsigned mode = descriptor >> 7;
    if (exceptionBits == 0 || bits + exceptionBits > elementBytes * 8 ||
        mode > GEOZL_PFOR_MODE_LIST)
      return 1;
    exceptionBytes = geozl_pfor_exc_bytes(exceptions, exceptionBits, mode);
  }

  const size_t bodyBytes = geozl_pfor_body_bytes(bits);
  if (exceptionBytes > SIZE_MAX - 2 ||
      bodyBytes > SIZE_MAX - 2 - exceptionBytes)
    return 1;
  const size_t bytes = 2 + exceptionBytes + bodyBytes;
  if (bytes > available)
    return 1;

  *info = (PforBlockInfo){bytes, exceptions};
  return 0;
}

uint8_t *cut_frames(const uint16_t *raw, size_t cells, size_t bands,
                    size_t side, size_t tile, int wholeCells, int u8,
                    Shape **shapes, size_t *nShapes) {
  const size_t per = side / tile, W = u8 ? 1 : 2;
  const size_t n = wholeCells ? cells : cells * bands * per * per;
  uint8_t *orig = page_alloc(cells * bands * side * side * W, NULL);
  Shape *s = calloc(n, sizeof(Shape));
  size_t o = 0, f = 0;
#define PUT(v)                                                                 \
  do {                                                                         \
    const uint16_t v_ = (v);                                                   \
    if (u8)                                                                    \
      orig[o] = (uint8_t)(v_ >> 4 > 255 ? 255 : v_ >> 4);                      \
    else                                                                       \
      memcpy(orig + o * 2, &v_, 2);                                            \
    ++o;                                                                       \
  } while (0)
  for (size_t c = 0; c < cells; ++c) {
    if (wholeCells) {
      s[f++] = (Shape){o, bands * side * side, side, (uint32_t)bands};
      for (size_t i = 0; i < bands * side * side; ++i)
        PUT(raw[c * bands * side * side + i]);
      continue;
    }
    for (size_t b = 0; b < bands; ++b)
      for (size_t ty = 0; ty < per; ++ty)
        for (size_t tx = 0; tx < per; ++tx) {
          s[f++] = (Shape){o, tile * tile, tile, 1};
          for (size_t y = 0; y < tile; ++y)
            for (size_t x = 0; x < tile; ++x)
              PUT(raw[((c * bands + b) * side + ty * tile + y) * side +
                      tx * tile + x]);
        }
  }
#undef PUT
  *shapes = s;
  *nShapes = n;
  return orig;
}

typedef struct {
  RangeFn fn;
  void *ctx;
  size_t begin, end;
  int failed;
} Job;

static void *run_job(void *p) {
  Job *j = p;
  j->failed = j->fn(j->ctx, j->begin, j->end) != 0;
  return NULL;
}

double cpu_best(RangeFn fn, void *ctx, size_t n, int threads, int reps) {
  Job jobs[256];
  pthread_t ids[256];
  if (threads > 256)
    threads = 256;
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const double t0 = now_seconds();
    for (int t = 0; t < threads; ++t) {
      jobs[t] = (Job){fn, ctx, n * (size_t)t / (size_t)threads,
                      n * (size_t)(t + 1) / (size_t)threads, 0};
      if (threads == 1)
        run_job(&jobs[t]);
      else
        pthread_create(&ids[t], NULL, run_job, &jobs[t]);
    }
    if (threads > 1)
      for (int t = 0; t < threads; ++t)
        pthread_join(ids[t], NULL);
    const double dt = now_seconds() - t0;
    for (int t = 0; t < threads; ++t)
      if (jobs[t].failed)
        return -1;
    if (dt < best)
      best = dt;
  }
  return best;
}

Planes plane_tasks(const Shape *shapes, size_t nShapes) {
  Planes p = {0};
  size_t n = 0;
  for (size_t f = 0; f < nShapes; ++f)
    n += shapes[f].planes;
  // Kernels run 4 planes per group; the padding planes have no rows.
  p.tasks = calloc((n + 3) / 4 * 4, sizeof(PlaneTask));
  for (size_t f = 0; f < nShapes; ++f) {
    const Shape *s = &shapes[f];
    const size_t planeElts = s->nbElts / s->planes;
    for (uint32_t k = 0; k < s->planes; ++k) {
      PlaneTask *t = &p.tasks[p.n++];
      *t = (PlaneTask){(uint32_t)(s->outOff + k * planeElts),
                       (uint32_t)s->width, (uint32_t)(planeElts / s->width)};
      if (t->width > p.maxWidth)
        p.maxWidth = t->width;
      if (t->rows > p.maxRows)
        p.maxRows = t->rows;
    }
  }
  p.n = (p.n + 3) / 4 * 4;
  return p;
}

int scan_dispatches(const Planes *p, GpuDispatch *out) {
  out[0] = (GpuDispatch){"scan_planes", (unsigned)(p->n / 4), 1, 128};
  if (p->maxWidth <= 256)
    return 1;
  out[1] = (GpuDispatch){"scan_rows", (unsigned)p->n, (p->maxRows + 3) / 4, 128};
  out[2] = (GpuDispatch){"scan_columns", (unsigned)p->n,
                         (p->maxWidth + 127) / 128, 128};
  return 3;
}

int report_gpu(const GpuTimes *gt, const GpuDispatch *d, int nd, double rawMB,
               double planS, const uint8_t *out, const uint8_t *want,
               size_t bytes) {
  printf("device    %s\n", gt->device);
  printf("passes   ");
  for (int i = 0; i < nd; ++i)
    printf(" %s %.2f ms", d[i].entry, gt->pass[i] * 1e3);
  if (gt->copy > 0)
    printf(", copy %.2f ms", gt->copy * 1e3);
  printf("\n");
  printf("gpu       %s kernel        %8.2f ms  %7.1f GB/s\n", gpu_backend_name,
         gt->kernel * 1e3, rawMB / 1e3 / gt->kernel);
  printf("gpu       %s + transfer    %8.2f ms  %7.1f GB/s  (plus host plan "
         "%.2f ms)\n",
         gpu_backend_name, gt->endToEnd * 1e3, rawMB / 1e3 / gt->endToEnd,
         planS * 1e3);
  size_t k = 0;
  while (k < bytes && out[k] == want[k])
    ++k;
  if (k == bytes) {
    printf("check     bit-identical to the source samples\n");
    return 0;
  }
  printf("check     MISMATCH at byte %zu (got %u, want %u)\n", k, out[k],
         want[k]);
  return 1;
}
