// Beyond the bound holding, what this pins down is that nothing about the grid
// is read from the tile. A grid that follows the data makes the same value
// rebuild two ways once a raster is cut differently, and no per-sample check on
// the error sees it.

#include "quant_log/decode_quant_log_kernel.h"
#include "quant_log/encode_quant_log_kernel.h"
#include "quant_log/quant_log_dtype.h"
#include "quant_log/quant_log_half.h"
#include "quant_log/quant_log_math.h"
#include "quant_log/quant_log_spec.h"

#include "quant_log_tiles.h"

#include <math.h>
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

static void the_bound_holds(void) {
  printf("the declared bound holds, every case and every tile\n");
  const char *rec[] = {"LOG:MAX_ERROR=5%", "LOG:MAX_ERROR=1%",
                       "LOG:MAX_ERROR=0.1%"};
  const double bound[] = {0.05, 0.01, 0.001};
  const char *all[QT_INT_N + QT_FLT_N];
  size_t na = 0;
  for (size_t i = 0; i < QT_INT_N; ++i)
    all[na++] = qt_int_tiles[i];
  for (size_t i = 0; i < QT_FLT_N; ++i)
    all[na++] = qt_flt_tiles[i];

  for (size_t d = 0; d < na; ++d) {
    qt_tile t = qt_make(all[d]);
    CHECK(t.data != NULL);
    for (size_t r = 0; r < 3; ++r) {
      void *st = alloc_for(t.dtype, t.n), *dec = alloc_for(t.dtype, t.n);
      if (trip(rec[r], t.dtype, t.data, st, dec, t.n, NULL) == 0) {
        const double w = worst_error(&t, dec, NULL);
        if (w > bound[r])
          printf("  %s %s: worst %.4e, bound %.4e\n", t.name, rec[r], w,
                 bound[r]);
        CHECK(w <= bound[r]);
      }
      free(st);
      free(dec);
    }
    qt_free(&t);
  }
}

// STORE=VALUES on a float, on the tiles that qualify for it.
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
  const char *names[] = {"s2_dn", "dem_i16", "humidity", "half", "wide"};
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
  qt_tile t = qt_make("s2_dn");
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
  const char *names[] = {"humidity", "s2_dn", "kelvin"};
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
}

static void the_parser_is_strict(void) {
  printf("the parser refuses what it cannot mean\n");
  quant_log_spec sp;
  char err[256];
  CHECK(quant_log_parse("LOG:MAX_ERROR=1%", &sp, err, sizeof err) == 0);
  CHECK(sp.rel_err == 0.01 && sp.store == QUANT_LOG_STORE_INDEX);
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

  // Parameters no encoder would write.
  quant_log_params bad;
  bad.flags = 0;
  bad.step = 0.0;
  float in = 1.0f;
  int32_t st = 0;
  float out = 0.0f;
  CHECK(quant_log_encode(&st, &in, &bad, QLOG_F32, 1) != 0);
  CHECK(quant_log_decode(&out, &st, &bad, QLOG_F32, 1) != 0);
  bad.step = 0.02;
  uint16_t u = 1, us = 0, uo = 0;
  CHECK(quant_log_encode(&us, &u, &bad, QLOG_U16, 1) != 0);
  CHECK(quant_log_decode(&uo, &us, &bad, QLOG_U16, 1) != 0);
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

int main(void) {
  the_bound_holds();
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
  the_refusals();
  a_forged_stream_stays_in_range();
  the_encoder_checks_its_own_work();
  report_streams();

  if (failures != 0) {
    printf("\n%d failed\n", failures);
    return 1;
  }
  printf("\nall passed\n");
  return 0;
}
