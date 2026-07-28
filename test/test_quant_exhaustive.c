// Walks every one of the 2^32 bit patterns a float32 can hold, for each recipe,
// and reports how many of them the codec cannot quantize inside the bound it
// declared. Fuzzing says nobody found a counterexample. This says there is not
// one.
//
// The methodology is from Fallin and Burtscher, "Lessons Learned on the Path to
// Guaranteeing the Error Bound in Lossy Quantizers", arXiv:2407.15037, who
// tested LC the same way and found that most published lossy compressors miss
// their bound on some values. Their Table 9 is the number this prints, the
// share of inputs a rounding pushes outside the bound, which for them ran from
// 0.12% to 11.16% depending on the dataset.
//
// A count above zero is not a failure. It is the share of values the encoder
// has to fall back on, and knowing it is the point. What would be a failure is
// a value that comes back non-finite, or one outside the range of its own type,
// since no fallback repairs that.
//
// Not part of make test-c, it takes minutes. Run it with make test-exhaustive.

#include "quant/decode_quant_kernel.h"
#include "quant/encode_quant_kernel.h"
#include "quant/quant_spec.h"

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH 8192

typedef struct {
  const char *recipe;
  double lo, hi; // the window a tile scan would have reported
  quant_spec sp;
  quant_params p;
  int usable;
} config;

typedef struct {
  const config *cfg;
  uint64_t begin, end, stride;
  uint64_t outside, nonfinite, broken, tested;
  uint64_t sum; // order independent, so the thread split does not change it
  double worst;
} job;

static double declared(const quant_spec *sp, double x) {
  switch (sp->curve) {
  case QUANT_CURVE_LOG:
    return sp->rel_err * fabs(x);
  case QUANT_CURVE_SQRT:
    return sp->shot_k * sqrt(sp->shot_a + sp->shot_b * fabs(x));
  default:
    return sp->abs_err;
  }
}

static void *walk(void *arg) {
  job *j = (job *)arg;
  const config *c = j->cfg;
  static const double vlo = -3.40282346638528860e38;
  static const double vhi = 3.40282346638528860e38;

  float src[BATCH], back[BATCH];
  int32_t idx[BATCH];

  for (uint64_t base = j->begin; base < j->end; base += BATCH) {
    const size_t n =
        (size_t)((j->end - base) < BATCH ? (j->end - base) : BATCH);
    for (size_t i = 0; i < n; ++i) {
      // A short run strides across the whole space rather than taking a prefix,
      // since the low bit patterns are all subnormal and would say nothing
      // about the rest.
      const uint32_t bits = (uint32_t)((base + i) * j->stride);
      memcpy(&src[i], &bits, sizeof(float));
    }
    if (quant_encode(idx, src, &c->p, Q_F32, n) ||
        quant_decode(back, idx, &c->p, Q_F32, n)) {
      j->broken += n;
      continue;
    }
    for (size_t i = 0; i < n; ++i) {
      // Every reconstruction folded into one number, order independent so the
      // thread count cannot change it. Two builds that disagree here disagree
      // bit for bit, which is what a frame written on one machine and read on
      // another cannot afford. Compare it across compilers, across flags and
      // across platforms.
      uint32_t rb;
      memcpy(&rb, &back[i], sizeof(rb));
      j->sum += (uint64_t)rb * 1099511628211ull + (uint64_t)(base + i);

      const double x = (double)src[i], y = (double)back[i];
      if (!isfinite(x))
        continue; // the nodata codec owns these
      // No fallback repairs a reconstruction that is not a number of the type
      // it is stored in, so these are failures rather than outliers.
      if (!isfinite(y) || y < vlo || y > vhi) {
        j->nonfinite++;
        continue;
      }
      // Shot noise is defined on non-negative data and the resolver refuses a
      // tile that has any, so a negative here is outside the domain rather
      // than a failure of the curve.
      if (c->sp.curve == QUANT_CURVE_SQRT && x < 0.0)
        continue;
      const double a = fabs(x);
      if (a != 0.0 && (a < c->lo || a > c->hi))
        continue; // outside the window the parameters were resolved for
      j->tested++;
      if (x == y)
        continue;
      const double b = declared(&c->sp, x);
      const double r = b > 0.0 ? fabs(x - y) / b : INFINITY;
      if (r > 1.0)
        j->outside++;
      if (r > j->worst)
        j->worst = r;
    }
  }
  return NULL;
}

static int run(config *c, int threads, uint64_t total, uint64_t stride) {
  char err[256] = {0};
  if (quant_spec_parse(c->recipe, &c->sp, err, sizeof(err)) != 0) {
    printf("  %-22s parse: %s\n", c->recipe, err);
    return 1;
  }
  if (quant_spec_resolve(&c->sp, Q_F32, c->lo, c->hi, 0, &c->p, err,
                         sizeof(err)) != 0) {
    printf("  %-22s refused, %s\n", c->recipe, err);
    return 0; // a refusal is an answer, not a failure
  }

  pthread_t th[64];
  job jb[64];
  const uint64_t span = total / (uint64_t)threads;
  for (int t = 0; t < threads; ++t) {
    jb[t] = (job){.cfg = c,
                  .begin = (uint64_t)t * span,
                  .end = (t == threads - 1) ? total : (uint64_t)(t + 1) * span,
                  .stride = stride};
    pthread_create(&th[t], NULL, walk, &jb[t]);
  }
  uint64_t outside = 0, nonfinite = 0, broken = 0, tested = 0, sum = 0;
  double worst = 0.0;
  for (int t = 0; t < threads; ++t) {
    pthread_join(th[t], NULL);
    outside += jb[t].outside;
    nonfinite += jb[t].nonfinite;
    broken += jb[t].broken;
    tested += jb[t].tested;
    sum += jb[t].sum;
    if (jb[t].worst > worst)
      worst = jb[t].worst;
  }

  const double pct = tested ? 100.0 * (double)outside / (double)tested : 0.0;
  printf("  %-22s tested %12" PRIu64 "  over %8" PRIu64 " (%.4f%%)"
         "  worst %.6f  sum %016" PRIx64,
         c->recipe, tested, outside, pct, worst, sum);
  if (nonfinite || broken)
    printf("  BROKEN nonfinite=%" PRIu64 " rejected=%" PRIu64, nonfinite,
           broken);
  printf("\n");
  return (nonfinite || broken) ? 1 : 0;
}

int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 8;
  // A fraction of the space, for a quick pass. Full run is the default.
  uint64_t total = argc > 2 ? strtoull(argv[2], NULL, 0) : (1ull << 32);
  if (threads < 1 || threads > 64)
    threads = 8;

  config cfgs[] = {
      {"abs:0.001", 1e-6, 1e3, {0}, {0}, 0},
      {"abs:0.5", 1e-3, 1e6, {0}, {0}, 0},
      {"rel:0.17%", 1.4012984643248171e-45, 3.40282346638528860e38, {0}, {0}, 0},
      {"rel:1%", 1.4012984643248171e-45, 3.40282346638528860e38, {0}, {0}, 0},
      {"rel:10.71%", 1.4012984643248171e-45, 3.40282346638528860e38, {0},
       {0}, 0},
      {"rel:1%", 1e-9, 1e9, {0}, {0}, 0},
      {"shot:a=100,b=1,k=0.5", 1.0, 1e5, {0}, {0}, 0},
      {"shot:a=4,b=1,k=0.5", 1.0, 1e4, {0}, {0}, 0},
  };

  const uint64_t stride = (1ull << 32) / (total ? total : 1);
  printf("float32, %" PRIu64 " of 4294967296 bit patterns%s, %d threads\n",
         total, stride > 1 ? " (strided)" : " (every one)", threads);
  int bad = 0;
  for (size_t i = 0; i < sizeof(cfgs) / sizeof(*cfgs); ++i)
    bad += run(&cfgs[i], threads, total, stride ? stride : 1);

  printf("\n%s\n", bad ? "BROKEN, a value came back unusable"
                       : "no value came back unusable");
  return bad != 0;
}