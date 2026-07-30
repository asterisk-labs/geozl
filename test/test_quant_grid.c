// Two things the error bound does not catch on its own.
//
// A frame can hold its declared bound at every sample and still hand back a
// negative reflectance, because the sqrt grid is anchored at -offset and the
// bound at zero is wide enough to cover the crossing. And two frames can each
// hold the bound while disagreeing with each other by twice it, because a grid
// cut against one tile is not the grid cut against another. Neither shows up in
// a test that only measures |x - x^| against the bound, so they get their own.

#include "quant/decode_quant_kernel.h" // quant_decode
#include "quant/encode_quant_kernel.h" // quant_scan
#include "quant/quant_dtype.h"
#include "quant/quant_spec.h"

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

// One round trip through the kernels the way an encoder does it, so the step a
// frame would be written with is the step this measures.
static int trip(const char *recipe, int dtype, const void *src, void *dec,
                size_t n, quant_params *pOut) {
  char err[256];
  quant_spec sp;
  if (quant_spec_parse(recipe, &sp, err, sizeof(err)) != 0)
    return -1;
  double lo = 0.0, hi = 0.0;
  int neg = 0;
  quant_scan(src, dtype, n, &lo, &hi, &neg);
  quant_params p;
  if (quant_spec_resolve(&sp, dtype, lo, hi, neg, &p, err, sizeof(err)) != 0)
    return -1;
  const size_t w = quant_width(dtype);
  void *idx = malloc(n * w);
  void *chk = malloc(n * w);
  if (idx == NULL || chk == NULL) {
    free(idx);
    free(chk);
    return -1;
  }
  const int fit = quant_fit(idx, chk, src, &sp, &p, dtype, n);
  int rc = -1;
  if (fit == 0 && quant_decode(dec, idx, &p, dtype, n) == 0)
    rc = 0;
  free(idx);
  free(chk);
  if (pOut != NULL)
    *pOut = p;
  return rc;
}

// Reflectance-like: non-negative, a hard floor of exact zeros where the footprint
// ends, and a shot bound, which is the one combination that used to come back
// negative.
static void nonnegative_stays_nonnegative(void) {
  printf("a tile with nothing negative decodes to nothing negative\n");
  enum { N = 4096 };
  static float src[N], dec[N];
  for (size_t i = 0; i < N; ++i)
    src[i] = i < 512 ? 0.0f : (float)(i % 1700) / 10000.0f;

  static const char *recipes[] = {"shot:a=9e-8,b=6e-5,k=0.5", "abs:0.0005",
                                  "rel:1%", "shot:a=0,b=1e-3,k=0.5",
                                  "shot:a=1,b=1,k=0.5"};
  for (size_t r = 0; r < sizeof(recipes) / sizeof(*recipes); ++r) {
    quant_params p;
    if (trip(recipes[r], Q_F32, src, dec, N, &p) != 0)
      continue; // a refusal is a legitimate answer, it just carries no samples
    CHECK((p.flags & QUANT_FLAG_NONNEGATIVE) != 0);
    size_t below = 0, zeros = 0, exact = 0;
    for (size_t i = 0; i < N; ++i) {
      if (dec[i] < 0.0f)
        ++below;
      if (src[i] == 0.0f) {
        ++zeros;
        if (dec[i] == 0.0f)
          ++exact;
      }
    }
    if (below != 0)
      printf("  %s emitted %zu negative samples\n", recipes[r], below);
    CHECK(below == 0);
    CHECK(exact == zeros); // and an exact zero comes back exact
  }
}

// The mirror of the above. A tile that does hold negatives must keep them, or
// the floor would be a guess about the data rather than a measurement of it.
static void negatives_survive(void) {
  printf("a tile that holds negatives keeps them\n");
  enum { N = 4096 };
  static int16_t src[N], dec[N];
  for (size_t i = 0; i < N; ++i)
    src[i] = (int16_t)((int)(i % 4000) - 420);

  static const char *recipes[] = {"abs:5", "abs:50", "rel:1%"};
  for (size_t r = 0; r < sizeof(recipes) / sizeof(*recipes); ++r) {
    quant_params p;
    if (trip(recipes[r], Q_I16, src, dec, N, &p) != 0)
      continue;
    CHECK((p.flags & QUANT_FLAG_NONNEGATIVE) == 0);
    int negatives = 0;
    for (size_t i = 0; i < N; ++i)
      if (dec[i] < 0)
        ++negatives;
    CHECK(negatives > 0);
  }
}

// The same raster cut two ways. Where the grid does not depend on the tile the
// two reconstructions have to agree exactly, not merely both hold the bound.
static void chunking_does_not_change_the_answer(void) {
  printf("cutting the raster differently does not change the reconstruction\n");
  enum { N = 4096 };
  static uint16_t src[N], a[N], b[N];
  for (size_t i = 0; i < N; ++i)
    src[i] = (uint16_t)(33000 + (i * 7919) % 21000); // all of it above 8/b

  // rel anchors on the type and the bound, so it must agree. abs on an integer
  // type takes its step straight from the bound, so it must agree too.
  static const char *recipes[] = {"rel:1%", "rel:0.5%", "abs:5", "abs:50"};
  for (size_t r = 0; r < sizeof(recipes) / sizeof(*recipes); ++r) {
    int ok = 1;
    for (size_t chunk = 256; chunk <= 1024; chunk *= 4) {
      uint16_t *dst = chunk == 256 ? a : b;
      for (size_t off = 0; off < N; off += chunk)
        if (trip(recipes[r], Q_U16, src + off, dst + off, chunk, NULL) != 0)
          ok = 0;
    }
    if (!ok)
      continue;
    size_t differ = 0;
    for (size_t i = 0; i < N; ++i)
      if (a[i] != b[i])
        ++differ;
    if (differ != 0)
      printf("  %s: %zu of %d samples differ between chunk 256 and 1024\n",
             recipes[r], differ, N);
    CHECK(differ == 0);
  }
}

// A declared range is what the grid is cut against, so the parameters must not
// move when the tile does, and a range that does not contain the tile has to be
// refused rather than quietly producing a grid for the wrong bound.
static void declared_range_pins_the_grid(void) {
  printf("a declared range pins the grid and rejects a tile it does not hold\n");
  char err[256];
  quant_spec sp;
  quant_params p1, p2;

  CHECK(quant_spec_parse("abs:0.05,max=400", &sp, err, sizeof(err)) == 0);
  CHECK(quant_spec_resolve(&sp, Q_F32, 1.0, 100.0, 0, &p1, err, sizeof(err)) ==
        0);
  CHECK(quant_spec_resolve(&sp, Q_F32, 1.0, 300.0, 0, &p2, err, sizeof(err)) ==
        0);
  CHECK(p1.step == p2.step); // two tiles, one grid
  CHECK(quant_spec_resolve(&sp, Q_F32, 1.0, 401.0, 0, &p1, err, sizeof(err)) !=
        0);

  CHECK(quant_spec_parse("rel:1%,min=33000", &sp, err, sizeof(err)) == 0);
  CHECK(quant_spec_resolve(&sp, Q_U16, 40000.0, 50000.0, 0, &p1, err,
                           sizeof(err)) == 0);
  CHECK(quant_spec_resolve(&sp, Q_U16, 35000.0, 60000.0, 0, &p2, err,
                           sizeof(err)) == 0);
  CHECK(p1.step == p2.step);
  CHECK(p1.offset == 33000.0);
  CHECK(p1.offset == p2.offset);
  CHECK(quant_spec_resolve(&sp, Q_U16, 32999.0, 60000.0, 0, &p1, err,
                           sizeof(err)) != 0);

  // Without a declaration the log anchor still must not follow the tile.
  CHECK(quant_spec_parse("rel:1%", &sp, err, sizeof(err)) == 0);
  CHECK(quant_spec_resolve(&sp, Q_U16, 40000.0, 50000.0, 0, &p1, err,
                           sizeof(err)) == 0);
  CHECK(quant_spec_resolve(&sp, Q_U16, 900.0, 50000.0, 0, &p2, err,
                           sizeof(err)) == 0);
  CHECK(p1.offset == p2.offset);
  CHECK(p1.step == p2.step);

  // Both parts, either order, and the ways of getting it wrong.
  CHECK(quant_spec_parse("rel:1%,min=10,max=900", &sp, err, sizeof(err)) == 0);
  CHECK(quant_spec_parse("rel:1%,max=900,min=10", &sp, err, sizeof(err)) == 0);
  CHECK(quant_spec_parse("shot:a=9,b=0.6,k=0.5,max=20000", &sp, err,
                         sizeof(err)) == 0);
  CHECK(quant_spec_parse("abs:1,max=0", &sp, err, sizeof(err)) != 0);
  CHECK(quant_spec_parse("abs:1,max=-5", &sp, err, sizeof(err)) != 0);
  CHECK(quant_spec_parse("abs:1,max=5,max=6", &sp, err, sizeof(err)) != 0);
  CHECK(quant_spec_parse("abs:1,min=9,max=2", &sp, err, sizeof(err)) != 0);
  CHECK(quant_spec_parse("abs:1,zzz=3", &sp, err, sizeof(err)) != 0);
  CHECK(quant_spec_parse("abs:1,max=", &sp, err, sizeof(err)) != 0);
  CHECK(quant_spec_parse("abs:1,", &sp, err, sizeof(err)) != 0);

  // And an undeclared recipe leaves them undeclared, since zero is a range a
  // caller could have meant.
  CHECK(quant_spec_parse("abs:1", &sp, err, sizeof(err)) == 0);
  CHECK(isnan(sp.decl_min) && isnan(sp.decl_max));
}

int main(void) {
  nonnegative_stays_nonnegative();
  negatives_survive();
  chunking_does_not_change_the_answer();
  declared_range_pins_the_grid();
  if (failures != 0) {
    printf("%d failed\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
