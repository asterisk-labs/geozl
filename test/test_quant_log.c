#include "quant_log/decode_quant_log_kernel.h"
#include "quant_log/encode_quant_log_kernel.h"
#include "quant_log/quant_log_dtype.h"
#include "quant_log/quant_log_half.h"
#include "quant_log/quant_log_math.h"
#include "quant_log/quant_log_spec.h"

#include "quant_log_tiles.h"

#include "quant_walk.h"

#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);                    \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static void *alloc_for(int dtype, size_t n) {
  return malloc(n * quant_log_width(dtype));
}

// One trip the way an encode binding would do it.
static int trip(const char *recipe, int dtype, const void *src, void *stream,
                void *dec, size_t n, quant_log_params *pOut) {
  char err[256];
  quant_log_spec sp;
  quant_log_stats sc;
  quant_log_params p;
  if (quant_log_parse(recipe, &sp, err, sizeof err) != 0)
    return -1;
  if (quant_log_scan(src, dtype, n, &sc) != 0)
    return -1;
  if (quant_log_resolve(&sp, dtype, &sc, &p, err, sizeof err) != 0)
    return -1;
  if (quant_log_encode(stream, src, &p, dtype, n) != 0)
    return -1;
  if (quant_log_decode(dec, stream, &p, dtype, n) != 0)
    return -1;
  if (pOut != NULL)
    *pOut = p;
  return 0;
}

static double worst_error(const qt_tile *t, const void *dec, size_t *skipped) {
  const double nmin =
      t->dtype > QLOG_LAST_INT ? quant_log_normal_min(t->dtype) : 0.0;
  double w = 0.0;
  size_t skip = 0;
  for (size_t i = 0; i < t->n; ++i) {
    const double x = qt_get(t->data, t->dtype, i);
    const double y = qt_get(dec, t->dtype, i);
    if (!isfinite(x) || x == 0.0 || x == y)
      continue;
    if (fabs(x) < nmin) {
      ++skip;
      continue;
    }
    const double e = fabs(y - x) / fabs(x);
    if (e > w)
      w = e;
  }
  if (skipped != NULL)
    *skipped = skip;
  return w;
}

static void store_values_holds_where_it_applies(void) {
  printf("STORE=VALUES holds where it applies and is refused where it does not\n");
  const char *names[] = {"kelvin", "utm", "reflectance", "humidity"};
  for (size_t d = 0; d < 4; ++d) {
    qt_tile t = qt_make(names[d]);
    void *st = alloc_for(t.dtype, t.n), *dec = alloc_for(t.dtype, t.n);
    char err[256];
    quant_log_spec sp;
    quant_log_stats sc;
    quant_log_params p;
    quant_log_parse("LOG:MAX_ERROR=1%,STORE=VALUES", &sp, err, sizeof err);
    quant_log_scan(t.data, t.dtype, t.n, &sc);
    if (quant_log_resolve(&sp, t.dtype, &sc, &p, err, sizeof err) == 0) {
      CHECK(quant_log_encode(st, t.data, &p, t.dtype, t.n) == 0);
      CHECK(quant_log_decode(dec, st, &p, t.dtype, t.n) == 0);
      const double w = worst_error(&t, dec, NULL);
      printf("  %-14s accepted, worst %.4e\n", t.name, w);
      CHECK(w <= 0.01);
    } else {
      printf("  %-14s refused, %s\n", t.name, err);
    }
    free(st);
    free(dec);
    qt_free(&t);
  }
}

// Integers below the crossover come back exact, because there the gap between
// levels is under one and the level nearest a whole number rounds back to it.
static void small_integers_come_back_exact(void) {
  printf("integers below the crossover come back exactly\n");
  uint16_t in[4096], st[4096], out[4096];
  for (int i = 0; i < 4096; ++i)
    in[i] = (uint16_t)i;
  quant_log_params p;
  CHECK(trip("LOG:MAX_ERROR=1%", QLOG_U16, in, st, out, 4096, &p) == 0);

  const double cross = 0.5 / (sqrt(1.01) - 1.0);
  int exact = 0, firstMoved = -1;
  for (int i = 0; i < 4096; ++i) {
    if (out[i] == in[i])
      ++exact;
    else if (firstMoved < 0)
      firstMoved = i;
  }
  printf("  the crossover sits at %.1f, the first integer to move is %d\n",
         cross, firstMoved);
  CHECK(firstMoved > (int)cross - 2);
  CHECK(exact > 100);
  for (int i = 0; i < (int)cross - 1; ++i)
    CHECK(out[i] == in[i]);
}

static void zero_and_nan_stay_zero(void) {
  printf("zero and NaN take index zero and come back as zero\n");
  float in[4] = {0.0f, -0.0f, 1.0f, 0.0f};
  in[3] = (float)NAN;
  int32_t st[4];
  float dec[4];
  CHECK(trip("LOG:MAX_ERROR=1%", QLOG_F32, in, st, dec, 4, NULL) == 0);
  CHECK(st[0] == 0 && st[1] == 0 && st[3] == 0 && st[2] != 0);
  CHECK(dec[0] == 0.0f && dec[1] == 0.0f && dec[3] == 0.0f);

  uint16_t di[3] = {0, 0, 700};
  uint16_t ds[3], dd[3];
  CHECK(trip("LOG:MAX_ERROR=1%", QLOG_U16, di, ds, dd, 3, NULL) == 0);
  CHECK(dd[0] == 0 && dd[1] == 0 && dd[2] != 0);
}

static void the_sign_survives(void) {
  printf("the sign of a sample survives the trip\n");
  const char *names[] = {"anomaly", "dem_i16"};
  for (int k = 0; k < 2; ++k) {
    qt_tile t = qt_make(names[k]);
    void *st = alloc_for(t.dtype, t.n), *dec = alloc_for(t.dtype, t.n);
    CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, t.data, st, dec, t.n, NULL) == 0);
    for (size_t i = 0; i < t.n; ++i) {
      const double x = qt_get(t.data, t.dtype, i);
      const double y = qt_get(dec, t.dtype, i);
      if (x == 0.0 || !isfinite(x))
        continue;
      CHECK((x < 0.0) == (y < 0.0));
    }
    free(st);
    free(dec);
    qt_free(&t);
  }
}

// The one that catches a grid anchored on the tile. Every sample keeps the
// stream element it had when a value far outside the range is dropped in.
static void the_grid_never_reads_the_tile(void) {
  printf("one outlier moves nothing else, at every case and width\n");
  const char *names[] = {"counts_u16", "dem_i16", "humidity", "half", "wide"};
  const double outlier[] = {65535.0, -32000.0, 3.0e38, 60000.0, 1.0e300};

  for (int k = 0; k < 5; ++k) {
    qt_tile t = qt_make(names[k]);
    const size_t w = quant_log_width(t.dtype);
    void *a = alloc_for(t.dtype, t.n), *b = alloc_for(t.dtype, t.n);
    void *dec = alloc_for(t.dtype, t.n), *moved = alloc_for(t.dtype, t.n);
    quant_log_params pa, pb;
    CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, t.data, a, dec, t.n, &pa) == 0);
    memcpy(moved, t.data, t.n * w);
    qt_put(moved, t.dtype, t.n / 2, outlier[k]);
    CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, moved, b, dec, t.n, &pb) == 0);

    CHECK(pa.step == pb.step);
    CHECK(pa.flags == pb.flags || t.dtype == QLOG_I16);
    size_t moves = 0;
    for (size_t i = 0; i < t.n; ++i)
      if (i != t.n / 2 &&
          memcmp((char *)a + i * w, (char *)b + i * w, w) != 0)
        ++moves;
    if (moves != 0)
      printf("  %s: an outlier moved %zu other elements\n", t.name, moves);
    CHECK(moves == 0);
    free(a);
    free(b);
    free(dec);
    free(moved);
    qt_free(&t);
  }
}

// Case 1 hands the stream back unchanged, so the decode is a byte copy.
static void an_integer_decode_is_a_byte_copy(void) {
  printf("an integer decode hands the stream back unchanged\n");
  qt_tile t = qt_make("counts_u16");
  const size_t w = quant_log_width(t.dtype);
  void *st = alloc_for(t.dtype, t.n), *dec = alloc_for(t.dtype, t.n);
  CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, t.data, st, dec, t.n, NULL) == 0);
  CHECK(memcmp(st, dec, t.n * w) == 0);
  free(st);
  free(dec);
  qt_free(&t);
}

// The table is a cache, so it has to give the same bits as the path that does
// not use it. Cutting the tile into single elements forces the other branch.
static void the_table_and_the_long_way_agree(void) {
  printf("the level table gives the same bits as the path without it\n");
  qt_tile t = qt_make("humidity");
  const size_t w = quant_log_width(t.dtype);
  void *st = alloc_for(t.dtype, t.n);
  void *a = alloc_for(t.dtype, t.n), *b = alloc_for(t.dtype, t.n);
  quant_log_params p;
  CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, t.data, st, a, t.n, &p) == 0);
  for (size_t i = 0; i < t.n; ++i)
    CHECK(quant_log_decode((char *)b + i * w, (const char *)st + i * w, &p,
                           t.dtype, 1) == 0);
  CHECK(memcmp(a, b, t.n * w) == 0);
  free(st);
  free(a);
  free(b);
  qt_free(&t);
}

// A reconstruction fed back in lands on the level it came from, or a tile
// compressed twice would drift.
static void a_second_pass_moves_nothing(void) {
  printf("a reconstruction fed back in lands on the level it came from\n");
  const char *names[] = {"humidity", "counts_u16", "kelvin"};
  for (int k = 0; k < 3; ++k) {
    qt_tile t = qt_make(names[k]);
    const size_t w = quant_log_width(t.dtype);
    void *s1 = alloc_for(t.dtype, t.n), *s2 = alloc_for(t.dtype, t.n);
    void *d1 = alloc_for(t.dtype, t.n), *d2 = alloc_for(t.dtype, t.n);
    quant_log_params p;
    CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, t.data, s1, d1, t.n, &p) == 0);
    CHECK(quant_log_encode(s2, d1, &p, t.dtype, t.n) == 0);
    CHECK(quant_log_decode(d2, s2, &p, t.dtype, t.n) == 0);
    CHECK(memcmp(s1, s2, t.n * w) == 0);
    CHECK(memcmp(d1, d2, t.n * w) == 0);
    free(s1);
    free(s2);
    free(d1);
    free(d2);
    qt_free(&t);
  }
}

// Nothing in the kernels reaches libm, so this is what says the series are as
// accurate as the resolver assumes when it cuts the step.
static void the_series_stay_inside_the_slack(void) {
  printf("the series are as accurate as the resolver assumes\n");
  const int dt[] = {QLOG_F16, QLOG_F32, QLOG_F64};
  const int anchor[] = {-24, -149, -1074};
  const double top[] = {16.0, 128.0, 1024.0};
  const char *nm[] = {"f16", "f32", "f64"};

  for (int k = 0; k < 3; ++k) {
    double w = 0.0;
    for (int i = 0; i <= 400000; ++i) {
      const double y = (double)anchor[k] +
                       (top[k] - (double)anchor[k]) * (double)i / 400000.0 -
                       1e-9;
      const double x = exp2(y);
      if (!isfinite(x) || x <= 0.0)
        continue;
      const double e = fabs(qlog_log2_over(x, anchor[k]) - (log2(x) - anchor[k]));
      if (e > w)
        w = e;
    }
    const double s = quant_log_slack(dt[k], 0);
    printf("  log2 %s worst %.3e, budget %.3e, %.1fx of room\n", nm[k], w, s,
           s / w);
    CHECK(w < s);
  }

  double w = 0.0;
  for (int i = 0; i <= 400000; ++i) {
    const double y = -1022.0 + 2046.0 * (double)i / 400000.0;
    const double ref = exp2(y);
    if (!isfinite(ref) || ref == 0.0)
      continue;
    const double e = fabs(qlog_exp2(y) - ref) / ref;
    if (e > w)
      w = e;
  }
  const double s = quant_log_slack(QLOG_F64, 0) * QLOG_LN2;
  printf("  exp2     worst %.3e, budget %.3e, %.1fx of room\n", w, s, s / w);
  CHECK(w < s);

  // The f64 index grid anchors at 2^-1074, so the bottom of it is subnormal and
  // the slack cannot hold there: the neighbours are a flat 2^-1074 apart and no
  // relative budget survives that. What the series still owes is a level within
  // one step of where exp2 puts it, which is what the walk below asserts. The
  // declared bound starts at the smallest normal, above all of this.
  const double sub = 4.9406564584124654e-324; // 2^-1074
  double wsub = 0.0;
  for (int i = 0; i <= 400000; ++i) {
    const double y = -1074.0 + 52.0 * (double)i / 400000.0;
    const double e = fabs(qlog_exp2(y) - exp2(y));
    if (e > wsub)
      wsub = e;
  }
  printf("  exp2     below the smallest normal, worst %.0f step of 2^-1074\n",
         wsub / sub);
  CHECK(wsub <= sub);
}

static void the_parser_is_strict(void) {
  printf("the parser refuses what it cannot mean\n");
  quant_log_spec sp;
  char err[256];
  CHECK(quant_log_parse("LOG:MAX_ERROR=1%", &sp, err, sizeof err) == 0);
  // The resolver chooses the dtype-specific default.
  CHECK(sp.rel_err == 0.01 && sp.store == QUANT_LOG_STORE_DEFAULT);
  CHECK(quant_log_parse("LOG:MAX_ERROR=0.5%,STORE=VALUES", &sp, err,
                        sizeof err) == 0);
  CHECK(sp.rel_err == 0.005 && sp.store == QUANT_LOG_STORE_VALUES);
  CHECK(quant_log_parse("LOG:STORE=INDEX,MAX_ERROR=2%", &sp, err, sizeof err) ==
        0);

  const char *bad[] = {NULL,
                       "",
                       "LOG:",
                       "LOG",
                       "log:MAX_ERROR=1%",
                       "LOG:MAX_ERROR=1",
                       "LOG:MAX_ERROR=%",
                       "LOG:MAX_ERROR=0%",
                       "LOG:MAX_ERROR=-1%",
                       "LOG:MAX_ERROR=100%",
                       "LOG:MAX_ERROR=1.0.0%",
                       "LOG:MAX_ERROR=1%,MAX_ERROR=2%",
                       "LOG:MAX_ERROR=1%,",
                       "LOG:MAX_ERROR=1%,STORE=BOTH",
                       "LOG:MIN=1,MAX_ERROR=1%",
                       "LINEAR:MAX_ERROR=1%"};
  for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i)
    CHECK(quant_log_parse(bad[i], &sp, err, sizeof err) != 0);
}

// Each refusal is a case where serving the request quietly would break the bound
// instead of reporting it.
static void the_refusals(void) {
  printf("each refusal reports rather than quietly breaking the bound\n");
  quant_log_spec sp;
  quant_log_params p;
  quant_log_stats sc;
  char err[256];
  memset(&sc, 0, sizeof sc);
  sc.minAbs = 1.0;
  sc.maxAbs = 1000.0;

  CHECK(quant_log_parse("LOG:MAX_ERROR=1%", &sp, err, sizeof err) == 0);
  CHECK(quant_log_resolve(&sp, 11, &sc, &p, err, sizeof err) != 0);
  CHECK(quant_log_resolve(&sp, -1, &sc, &p, err, sizeof err) != 0);

  // Under the rounding of the output width there is nothing left for the grid.
  CHECK(quant_log_parse("LOG:MAX_ERROR=0.001%", &sp, err, sizeof err) == 0);
  CHECK(quant_log_resolve(&sp, QLOG_F16, &sc, &p, err, sizeof err) != 0);
  printf("  f16 too tight: %s\n", err);
  CHECK(quant_log_resolve(&sp, QLOG_F32, &sc, &p, err, sizeof err) == 0);
  CHECK(quant_log_parse("LOG:MAX_ERROR=0.000001%", &sp, err, sizeof err) == 0);
  CHECK(quant_log_resolve(&sp, QLOG_F32, &sc, &p, err, sizeof err) != 0);
  printf("  f32 too tight: %s\n", err);

  // A float below the crossover cannot carry a whole-number reconstruction.
  CHECK(quant_log_parse("LOG:MAX_ERROR=1%,STORE=VALUES", &sp, err, sizeof err) ==
        0);
  sc.minAbs = 0.4;
  CHECK(quant_log_resolve(&sp, QLOG_F32, &sc, &p, err, sizeof err) != 0);
  printf("  too small: %s\n", err);

  // And one past what the type carries as a whole number.
  sc.minAbs = 1000.0;
  sc.maxAbs = 3.0e7;
  CHECK(quant_log_resolve(&sp, QLOG_F32, &sc, &p, err, sizeof err) != 0);
  printf("  too large: %s\n", err);
  sc.maxAbs = 1.0e6;
  CHECK(quant_log_resolve(&sp, QLOG_F32, &sc, &p, err, sizeof err) == 0);
}

// There is no decode binding yet, so the kernels are the only thing between a
// frame and the transform. Every field the header carries gets forged here.
static void a_forged_header_is_refused(void) {
  printf("the kernels refuse a parameter block the resolver cannot produce\n");
  enum { N = 64 };
  static float f[N], df[N];
  static int32_t sf[N];
  static uint16_t u[N], du[N];
  quant_log_params p;
  memset(&p, 0, sizeof p);

  const double bad[] = {0.0, -1.0, -0.02, HUGE_VAL, -HUGE_VAL, NAN};
  for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
    p.flags = 0;
    p.step = bad[i];
    CHECK(quant_log_encode(sf, f, &p, QLOG_F32, N) != 0);
    CHECK(quant_log_decode(df, sf, &p, QLOG_F32, N) != 0);
    p.flags = QUANT_LOG_FLAG_STORE_VALUES;
    CHECK(quant_log_encode(du, u, &p, QLOG_U16, N) != 0);
    CHECK(quant_log_decode(du, u, &p, QLOG_U16, N) != 0);
  }

  // An integer frame without the flag is one this codec never writes.
  p.step = 0.02;
  p.flags = 0;
  CHECK(quant_log_encode(du, u, &p, QLOG_U16, N) != 0);
  CHECK(quant_log_decode(du, u, &p, QLOG_U16, N) != 0);

  // Flag bits the format does not define.
  for (unsigned bit = 2; bit < 8; ++bit) {
    p.flags = (unsigned char)(QUANT_LOG_FLAG_STORE_VALUES | (1u << bit));
    CHECK(quant_log_encode(sf, f, &p, QLOG_F32, N) != 0);
    CHECK(quant_log_decode(df, sf, &p, QLOG_F32, N) != 0);
  }

  p.flags = QUANT_LOG_FLAG_STORE_VALUES;
  CHECK(quant_log_encode(du, u, &p, 11, N) != 0);
  CHECK(quant_log_decode(du, u, &p, -1, N) != 0);
}

// A forged stream cannot walk the reconstruction out of the output type, and on
// a tile marked non-negative it cannot produce a negative either.
static void a_forged_stream_stays_in_range(void) {
  printf("a forged stream stays inside the output type\n");
  quant_log_spec sp;
  quant_log_params p;
  quant_log_stats sc;
  char err[256];
  memset(&sc, 0, sizeof sc);
  sc.minAbs = 1000.0;
  sc.maxAbs = 1.0e6;

  const int32_t forged[] = {2147483647, -2147483647, 1000000, -1000000, 1, -1, 0};
  const size_t n = sizeof forged / sizeof *forged;
  float out[8];

  const char *rec[] = {"LOG:MAX_ERROR=1%", "LOG:MAX_ERROR=1%,STORE=VALUES"};
  for (int r = 0; r < 2; ++r) {
    CHECK(quant_log_parse(rec[r], &sp, err, sizeof err) == 0);
    for (int neg = 0; neg < 2; ++neg) {
      sc.anyNegative = neg;
      CHECK(quant_log_resolve(&sp, QLOG_F32, &sc, &p, err, sizeof err) == 0);
      CHECK(quant_log_decode(out, forged, &p, QLOG_F32, n) == 0);
      for (size_t i = 0; i < n; ++i) {
        CHECK(isfinite(out[i]));
        if (!neg)
          CHECK(out[i] >= 0.0f);
      }
    }
  }
}

// The encoder's own check, the way an encode binding would run it.
static void the_encoder_checks_its_own_work(void) {
  printf("the encoder measures its own round trip against the bound\n");
  const char *all[QT_INT_N + QT_FLT_N];
  size_t na = 0;
  for (size_t i = 0; i < QT_INT_N; ++i)
    all[na++] = qt_int_tiles[i];
  for (size_t i = 0; i < QT_FLT_N; ++i)
    all[na++] = qt_flt_tiles[i];

  for (size_t d = 0; d < na; ++d) {
    qt_tile t = qt_make(all[d]);
    void *st = alloc_for(t.dtype, t.n), *dec = alloc_for(t.dtype, t.n);
    quant_log_spec sp;
    quant_log_stats sc;
    char err[256];
    CHECK(quant_log_parse("LOG:MAX_ERROR=1%", &sp, err, sizeof err) == 0);
    CHECK(trip("LOG:MAX_ERROR=1%", t.dtype, t.data, st, dec, t.n, NULL) == 0);
    double worst = 0.0;
    size_t skipped = 0;
    CHECK(quant_log_verify(t.data, dec, &sp, t.dtype, t.n, &worst, &skipped) ==
          0);
    CHECK(worst <= 1.0);
    quant_log_scan(t.data, t.dtype, t.n, &sc);
    printf("  %-14s used %.6f of the budget, %zu below the smallest normal\n",
           t.name, worst, skipped);
    CHECK((skipped != 0) == (sc.anySubnormal != 0));
    free(st);
    free(dec);
    qt_free(&t);
  }
}

// The encoder caches the value grid while the table is small enough and builds
// each level on the spot when it is not. Both bounds below walk every uint16
// against the level spec.md states, worked out here instead of read from the
// kernel, so the two paths landing on different levels shows up as a wrong one.
static double spec_level(double a, double step, int dtype) {
  const double cap = quant_log_value_hi(dtype);
  const double top = quant_log_value_top(step, dtype);
  double j = qlog_log2_over(a, 0) / step + 0.5;
  if (!(j > 0.0))
    j = 0.0;
  else if (!(j < top))
    j = top;
  const double v = qlog_value_level((double)(int64_t)j, step);
  return v > cap ? cap : v;
}

static void the_value_grid_matches_the_spec(void) {
  printf("the value grid lands where the spec says, with and without the table\n");
  const char *rec[] = {"LOG:MAX_ERROR=1%", "LOG:MAX_ERROR=0.001%"};
  static uint16_t in[65536], st[65536];
  for (int i = 0; i < 65536; ++i)
    in[i] = (uint16_t)i;

  for (int r = 0; r < 2; ++r) {
    quant_log_spec sp;
    quant_log_stats sc;
    quant_log_params p;
    char err[256];
    CHECK(quant_log_parse(rec[r], &sp, err, sizeof err) == 0);
    CHECK(quant_log_scan(in, QLOG_U16, 65536, &sc) == 0);
    CHECK(quant_log_resolve(&sp, QLOG_U16, &sc, &p, err, sizeof err) == 0);
    CHECK(quant_log_encode(st, in, &p, QLOG_U16, 65536) == 0);
    size_t wrong = 0;
    for (int i = 1; i < 65536; ++i)
      if ((double)st[i] != spec_level((double)in[i], p.step, QLOG_U16))
        ++wrong;
    if (wrong != 0)
      printf("  %s: %zu of 65535 off the level the spec puts down\n", rec[r],
             wrong);
    CHECK(wrong == 0);
    CHECK(st[0] == 0);
  }
}

// u32, i32, u64 and i64 have no exhaustive walk. Every power of two, both sides
// of it, the limits of the type, and a stride across the rest.
static void wide_put(void *b, int dtype, size_t i, uint64_t v) {
  if (quant_log_width(dtype) == 4) {
    const uint32_t t = (uint32_t)v;
    memcpy((char *)b + i * 4, &t, 4);
  } else {
    memcpy((char *)b + i * 8, &v, 8);
  }
}

static double wide_get(const void *b, int dtype, size_t i) {
  if (quant_log_width(dtype) == 4) {
    uint32_t t;
    memcpy(&t, (const char *)b + i * 4, 4);
    return dtype == QLOG_U32 ? (double)t : (double)(int32_t)t;
  }
  uint64_t t;
  memcpy(&t, (const char *)b + i * 8, 8);
  return dtype == QLOG_U64 ? (double)t : (double)(int64_t)t;
}

static void the_wide_integers_hold(void) {
  printf("the wide integer types hold the bound and keep the sign\n");
  const char *rec[] = {"LOG:MAX_ERROR=5%", "LOG:MAX_ERROR=1%",
                       "LOG:MAX_ERROR=0.01%"};
  const double bound[] = {0.05, 0.01, 0.0001};
  const int dt[] = {QLOG_U32, QLOG_I32, QLOG_U64, QLOG_I64};
  enum { N = 200000 };
  static unsigned char src[N * 8], st[N * 8], bk[N * 8];

  for (int k = 0; k < 4; ++k) {
    const int signd = dt[k] == QLOG_I32 || dt[k] == QLOG_I64;
    const int bits = quant_log_width(dt[k]) == 4 ? 32 : 64;
    const int span = signd ? bits - 1 : bits;
    const uint64_t mask = span == 64 ? ~(uint64_t)0
                                     : (((uint64_t)1 << span) - 1);
    size_t n = 0;
    for (int e = 0; e < span; ++e) {
      const uint64_t v = (uint64_t)1 << e;
      wide_put(src, dt[k], n++, v);
      wide_put(src, dt[k], n++, v - 1);
      wide_put(src, dt[k], n++, v + 1);
      if (signd) {
        wide_put(src, dt[k], n++, (uint64_t)(-(int64_t)v));
        wide_put(src, dt[k], n++, (uint64_t)(1 - (int64_t)v));
      }
    }
    wide_put(src, dt[k], n++, mask);
    if (signd)
      wide_put(src, dt[k], n++, (uint64_t)(-(int64_t)mask - 1));

    uint64_t g = 0x9E3779B97F4A7C15ull;
    while (n < N) {
      g ^= g << 13;
      g ^= g >> 7;
      g ^= g << 17;
      const uint64_t v = (g >> (g % (unsigned)span)) & mask;
      wide_put(src, dt[k], n++, signd && (g & 1) ? (uint64_t)(-(int64_t)v) : v);
    }

    for (int r = 0; r < 3; ++r) {
      quant_log_params p;
      if (trip(rec[r], dt[k], src, st, bk, n, &p) != 0) {
        printf("  dtype %d refused %s\n", dt[k], rec[r]);
        ++failures;
        continue;
      }
      double worst = 0.0;
      for (size_t i = 0; i < n; ++i) {
        const double x = wide_get(src, dt[k], i), y = wide_get(bk, dt[k], i);
        if (x == 0.0) {
          CHECK(y == 0.0);
          continue;
        }
        CHECK((x < 0.0) == (y < 0.0));
        const double e = fabs(y - x) / fabs(x);
        if (e > worst)
          worst = e;
      }
      if (worst > bound[r])
        printf("  dtype %d %s: worst %.4e over %.4e\n", dt[k], rec[r], worst,
               bound[r]);
      CHECK(worst <= bound[r]);
    }
  }
}

// Case 2 is the case no exhaustive walk covers. On a half the band the resolver
// accepts is small enough to walk whole.
static void store_values_on_a_half_is_walked(void) {
  printf("STORE=VALUES on a half, every value of the band it accepts\n");
  const char *rec[] = {"LOG:MAX_ERROR=5%,STORE=VALUES",
                       "LOG:MAX_ERROR=1%,STORE=VALUES",
                       "LOG:MAX_ERROR=0.1%,STORE=VALUES"};
  const double bound[] = {0.05, 0.01, 0.001};
  static uint16_t in[65536], bk[65536];
  static int16_t st[65536];

  for (int r = 0; r < 3; ++r) {
    const double cross = 0.5 / (sqrt(1.0 + bound[r]) - 1.0);
    size_t n = 0;
    for (int i = 0; i < 65536; ++i) {
      const float v = quant_log_half_to_float((uint16_t)i);
      const double a = fabs((double)v);
      if (a >= cross && a <= 1024.0)
        in[n++] = (uint16_t)i;
    }
    CHECK(n > 0);
    quant_log_spec sp;
    quant_log_stats sc;
    quant_log_params p;
    char err[256];
    CHECK(quant_log_parse(rec[r], &sp, err, sizeof err) == 0);
    CHECK(quant_log_scan(in, QLOG_F16, n, &sc) == 0);
    CHECK(quant_log_resolve(&sp, QLOG_F16, &sc, &p, err, sizeof err) == 0);
    CHECK(quant_log_encode(st, in, &p, QLOG_F16, n) == 0);
    CHECK(quant_log_decode(bk, st, &p, QLOG_F16, n) == 0);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double x = (double)quant_log_half_to_float(in[i]);
      const double y = (double)quant_log_half_to_float(bk[i]);
      const double e = fabs(y - x) / fabs(x);
      if (e > worst)
        worst = e;
    }
    printf("  %-32s %zu values, worst %.4e of %.4e\n", rec[r], n, worst,
           bound[r]);
    CHECK(worst <= bound[r]);
  }
}

// What the tiles cost the codecs behind this one. Reported, not asserted.
static void report_streams(void) {
  printf("\n  tile           type   case              stream range\n");
  const char *all[QT_INT_N + QT_FLT_N];
  size_t na = 0;
  for (size_t i = 0; i < QT_INT_N; ++i)
    all[na++] = qt_int_tiles[i];
  for (size_t i = 0; i < QT_FLT_N; ++i)
    all[na++] = qt_flt_tiles[i];

  for (size_t d = 0; d < na; ++d) {
    qt_tile t = qt_make(all[d]);
    void *st = alloc_for(t.dtype, t.n), *dec = alloc_for(t.dtype, t.n);
    quant_log_params p;
    if (trip("LOG:MAX_ERROR=1%", t.dtype, t.data, st, dec, t.n, &p) == 0) {
      double lo = 0.0, hi = 0.0;
      for (size_t i = 0; i < t.n; ++i) {
        double v;
        if (t.dtype <= QLOG_LAST_INT)
          v = qt_get(st, t.dtype, i);
        else if (t.dtype == QLOG_F16)
          v = ((int16_t *)st)[i];
        else if (t.dtype == QLOG_F32)
          v = ((int32_t *)st)[i];
        else
          v = (double)((int64_t *)st)[i];
        if (i == 0)
          lo = hi = v;
        if (v < lo)
          lo = v;
        if (v > hi)
          hi = v;
      }
      const int values = (p.flags & QUANT_LOG_FLAG_STORE_VALUES) != 0;
      printf("  %-14s %-6s %-18s %.0f .. %.0f\n", t.name,
             t.dtype <= QLOG_LAST_INT  ? "int"
             : t.dtype == QLOG_F16     ? "f16"
             : t.dtype == QLOG_F32     ? "f32"
                                       : "f64",
             t.dtype <= QLOG_LAST_INT ? "1 reconstruction"
             : values                 ? "2 reconstruction"
                                      : "3 index",
             lo, hi);
    }
    free(st);
    free(dec);
    qt_free(&t);
  }
}


static void the_parser_takes_only_decimals(void) {
  puts("the parser takes only decimals");

  static const struct {
    const char *recipe;
    int ok;
  } cases[] = {
      {"LOG:MAX_ERROR=1%", 1},
      {"LOG:MAX_ERROR=0.5%", 1},
      {"LOG:MAX_ERROR=1e-3%", 1},
      {"LOG:MAX_ERROR=1E+1%", 1},
      {"LOG:MAX_ERROR=.5%", 1},
      {"LOG:MAX_ERROR=1%,STORE=VALUES", 1},
      {"LOG:MAX_ERROR=1%,STORE=INDEX", 1},
      {"LOG:MAX_ERROR=1", 0},   // a bound of one is not what anybody means
      {"LOG:MAX_ERROR=0%", 0},
      {"LOG:MAX_ERROR=100%", 0},
      {"LOG:MAX_ERROR=-1%", 0},
      {"LOG:MAX_ERROR=inf%", 0},
      {"LOG:MAX_ERROR=nan%", 0},
      {"LOG:MAX_ERROR=1.0.0%", 0},
      {"LOG:MAX_ERROR=1e%", 0},
      {"LOG:MAX_ERROR=1_0%", 0},
      {"LOG:MAX_ERROR=1e999%", 0},
      {"LOG:MAX_ERROR=1% ", 0},
      {"LOG:MAX_ERROR=1%,MAX_ERROR=2%", 0},
      {"LOG:MAX_ERROR=1%,", 0},
      {"LOG:STORE=VALUES", 0},
      {"LOG:", 0},
      {"LINEAR:MAX_ERROR=1", 0},
  };

  char err[256];
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    quant_log_spec sp;
    const int got = quant_log_parse(cases[i].recipe, &sp, err, sizeof(err)) == 0;
    if (got != cases[i].ok)
      printf("  \"%s\": got %d, expected %d\n", cases[i].recipe, got,
             cases[i].ok);
    CHECK(got == cases[i].ok);
  }
}

// Skipped where no comma locale is installed, which is most containers.
static void a_comma_locale_reads_the_same_recipe(void) {
  puts("a comma locale reads the same recipe");
  static const char *names[] = {"de_DE.UTF-8", "fr_FR.UTF-8", "es_PE.UTF-8",
                                "de_DE", "fr_FR"};
  const char *got = NULL;
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && got == NULL; ++i)
    got = setlocale(LC_NUMERIC, names[i]);

  if (got == NULL) {
    puts("  no comma locale installed, skipped");
    return;
  }
  printf("  under %s\n", got);

  quant_log_spec sp;
  char err[256];
  CHECK(quant_log_parse("LOG:MAX_ERROR=0.5%", &sp, err, sizeof(err)) == 0);
  CHECK(sp.rel_err == 0.005);
  // The comma separates keys, so it never reaches the number.
  CHECK(quant_log_parse("LOG:MAX_ERROR=0,5%", &sp, err, sizeof(err)) != 0);

  setlocale(LC_NUMERIC, "C");
}

// Found by the fuzzer on int16 at 0.002%. Folding the grid onto the maximum
// rather than the magnitude brought the most negative value back one short.
static void the_most_negative_value_survives(void) {
  puts("the most negative value of a signed type survives");

  static const struct {
    const char *what;
    int dtype;
    int64_t lowest;
  } cases[] = {
      {"int8", QLOG_I8, INT8_MIN},
      {"int16", QLOG_I16, INT16_MIN},
      {"int32", QLOG_I32, INT32_MIN},
      {"int64", QLOG_I64, INT64_MIN},
  };
  static const char *recipes[] = {"LOG:MAX_ERROR=0.002%", "LOG:MAX_ERROR=1%",
                                  "LOG:MAX_ERROR=5%"};

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    for (size_t r = 0; r < sizeof(recipes) / sizeof(recipes[0]); ++r) {
      int64_t in[4];
      in[0] = cases[c].lowest;
      in[1] = cases[c].lowest + 1;
      in[2] = -1;
      in[3] = 1;

      unsigned char src[4 * 8], enc[4 * 8], dec[4 * 8];
      const size_t w = quant_log_width(cases[c].dtype);
      for (size_t i = 0; i < 4; ++i)
        memcpy(src + i * w, &in[i], w); // little endian, which the tests assume

      char err[256];
      quant_log_spec sp;
      quant_log_params p;
      quant_log_stats sc;
      CHECK(quant_log_parse(recipes[r], &sp, err, sizeof(err)) == 0);
      CHECK(quant_log_scan(src, cases[c].dtype, 4, &sc) == 0);
      if (quant_log_resolve(&sp, cases[c].dtype, &sc, &p, err, sizeof(err)) != 0)
        continue;
      CHECK(quant_log_encode(enc, src, &p, cases[c].dtype, 4) == 0);
      CHECK(quant_log_decode(dec, enc, &p, cases[c].dtype, 4) == 0);

      double worst = 0.0;
      size_t skipped = 0;
      CHECK(quant_log_verify(src, dec, &sp, cases[c].dtype, 4, &worst,
                             &skipped) == 0);
      if (worst > 1.0)
        printf("  %s %s: worst %.6f of the bound\n", cases[c].what, recipes[r],
               worst);
      CHECK(worst <= 1.0);
    }
  }
}

// A frame that declares the floor has to get it whatever the element type is.
// The float paths always applied it and the integer copy did not, so the flag
// meant two things inside one codec.
static void the_floor_reaches_an_integer_frame(void) {
  puts("the floor reaches an integer frame");

  const int16_t stream[6] = {-3000, -1, 0, 1, 500, 32767};
  int16_t out[6];
  quant_log_params p = {QUANT_LOG_FLAG_NONNEGATIVE | QUANT_LOG_FLAG_STORE_VALUES,
                        0.0144};
  CHECK(quant_log_decode(out, stream, &p, QLOG_I16, 6) == 0);
  for (size_t i = 0; i < 6; ++i)
    CHECK(out[i] >= 0);
  CHECK(out[3] == 1 && out[4] == 500 && out[5] == 32767);

  // Without the flag the same stream comes back untouched.
  p.flags = QUANT_LOG_FLAG_STORE_VALUES;
  CHECK(quant_log_decode(out, stream, &p, QLOG_I16, 6) == 0);
  CHECK(out[0] == -3000 && out[1] == -1);

  // An unsigned type cannot hold a negative, so the flag changes nothing there.
  const uint16_t ustream[4] = {0, 1, 500, 65535};
  uint16_t uout[4];
  p.flags = QUANT_LOG_FLAG_NONNEGATIVE | QUANT_LOG_FLAG_STORE_VALUES;
  CHECK(quant_log_decode(uout, ustream, &p, QLOG_U16, 4) == 0);
  CHECK(memcmp(uout, ustream, sizeof(ustream)) == 0);
}


// Every value of a type through the grid, one at a time. The small domains are
// walked whole; f32 goes on a stride unless GEOZL_EXHAUSTIVE is set.
//
// anyNegative is set so the decoder floors at the type minimum rather than at
// zero, which puts the negative half of the domain through the grid instead of
// through the clamp.
static void every_value_holds_the_bound(void) {
  static const char *const recipes[] = {"LOG:MAX_ERROR=5%", "LOG:MAX_ERROR=1%",
                                        "LOG:MAX_ERROR=0.1%"};
  static const double bounds[] = {0.05, 0.01, 0.001};
  static uint8_t in[GEOZL_WALK_BLK * 4], st[GEOZL_WALK_BLK * 4];
  static uint8_t out[GEOZL_WALK_BLK * 4];
  const uint64_t step = geozl_walk_step();
  printf("every value of a type holds the bound, f32 %s\n",
         step == 1 ? "whole" : "on a stride");

  for (size_t t = 0; t < GEOZL_WALK_NTYPES; ++t) {
    const int dt = geozl_walk_types[t].dtype;
    const double nmin = quant_log_normal_min(dt);
    for (size_t r = 0; r < sizeof(bounds) / sizeof(bounds[0]); ++r) {
      char err[256];
      quant_log_spec sp;
      quant_log_stats sc;
      quant_log_params p;
      memset(&sc, 0, sizeof sc);
      sc.anyNegative = 1;
      sc.minAbs = 1.0;
      sc.maxAbs = 1.0;
      CHECK(quant_log_parse(recipes[r], &sp, err, sizeof err) == 0);
      if (quant_log_resolve(&sp, dt, &sc, &p, err, sizeof err) != 0)
        continue; // a refusal is an answer, and the refusals have their own case

      double worst = 0.0;
      uint64_t counted = 0, over = 0, sum = 0, at = 0;
      const uint64_t domain = geozl_walk_types[t].domain;
      const uint64_t span = dt == GEOZL_DT_F32 ? step : 1;
      for (uint64_t base = 0; base < domain; base += GEOZL_WALK_BLK * span) {
        const size_t n = geozl_walk_fill(in, dt, base, domain, span);
        CHECK(quant_log_encode(st, in, &p, dt, n) == 0);
        CHECK(quant_log_decode(out, st, &p, dt, n) == 0);
        for (size_t i = 0; i < n; ++i) {
          const uint64_t pos = base + (uint64_t)i * span;
          GEOZL_FOLD(sum, geozl_walk_bits(out, dt, i), pos);
          const double x = geozl_walk_get(in, dt, i);
          const double y = geozl_walk_get(out, dt, i);
          if (!isfinite(x))
            continue; // the nodata codec owns these
          if (x == 0.0) {
            CHECK(y == 0.0);
            continue;
          }
          if (fabs(x) < nmin)
            continue; // below the smallest normal no grid reaches
          ++counted;
          const double e = fabs(y - x) / fabs(x);
          if (e > worst) {
            worst = e;
            at = pos;
          }
          if (e > bounds[r])
            ++over;
        }
      }
      printf("  %-4s %-20s %10llu values, worst %.4f of the bound, sum %016llx\n",
             geozl_walk_types[t].name, recipes[r],
             (unsigned long long)counted, worst / bounds[r],
             (unsigned long long)sum);
      if (over != 0) {
        printf("    FAIL %llu over the bound, worst at 0x%llx\n",
               (unsigned long long)over, (unsigned long long)at);
        ++failures;
      }
    }
  }
}

int main(void) {
  store_values_holds_where_it_applies();
  small_integers_come_back_exact();
  zero_and_nan_stay_zero();
  the_sign_survives();
  the_grid_never_reads_the_tile();
  an_integer_decode_is_a_byte_copy();
  the_table_and_the_long_way_agree();
  a_second_pass_moves_nothing();
  the_series_stay_inside_the_slack();
  the_parser_is_strict();
  the_parser_takes_only_decimals();
  a_comma_locale_reads_the_same_recipe();
  the_refusals();
  a_forged_header_is_refused();
  a_forged_stream_stays_in_range();
  the_encoder_checks_its_own_work();
  the_value_grid_matches_the_spec();
  the_wide_integers_hold();
  the_most_negative_value_survives();
  the_floor_reaches_an_integer_frame();
  store_values_on_a_half_is_walked();
  every_value_holds_the_bound();
  report_streams();
  if (failures != 0) {
    printf("%d failed\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
