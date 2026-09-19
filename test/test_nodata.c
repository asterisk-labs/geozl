#include "nodata/decode_nodata_kernel.h"
#include "nodata/encode_nodata_kernel.h"
#include "planar/encode_planar_kernel.h"

#include "common/half.h"
#include "geozl/dtype.h"
#include "lossy/lossy_recipe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                 \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define ROWS 64
#define COLS 96
#define N (ROWS * COLS)

static uint8_t mask[N];
static uint32_t src32[N], filled32[N], back32[N];

// A NaN carries 22 payload bits in f32. Storing the pattern rather than a
// canonical NaN is what keeps the round trip lossless, so the test uses one
// that no hardware operation would produce.
#define ODD_NAN 0x7FC0BEEFu

static void build_tile(void) {
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      const float v = 285.0f + 0.5f * (float)c + 0.25f * (float)r;
      uint32_t bits;
      memcpy(&bits, &v, sizeof(bits));
      src32[r * COLS + c] = bits;
    }
  }
  // A coherent blob plus a whole trailing row, which exercises the fill path
  // that has no valid sample to its left and has to reach the row above.
  for (int r = 0; r < ROWS; ++r)
    for (int c = 0; c < COLS; ++c)
      if ((r - 20) * (r - 20) + (c - 30) * (c - 30) < 150 || r == ROWS - 1)
        src32[r * COLS + c] = ODD_NAN;
}

static void test_nan_round_trip(void) {
  uint64_t pattern = 0;
  CHECK(nodata_find_nan(&pattern, src32, N, 4) == 1);
  CHECK(pattern == ODD_NAN);

  nodata_mark_value(mask, src32, N, 4, pattern);

  nodata_fill(filled32, src32, mask, COLS, N, 4);
  nodata_restore(back32, filled32, mask, N, 4, pattern);
  CHECK(memcmp(back32, src32, sizeof src32) == 0);

  // Nothing the fill wrote may survive, a reader only ever sees the pattern.
  for (size_t i = 0; i < N; ++i)
    if (mask[i] == GEOZL_NODATA_INVALID && back32[i] != ODD_NAN) {
      CHECK(0);
      break;
    }
}

// An infinity is a measurement, not a hole, and must not be swept up.
static void test_infinity_is_a_value(void) {
  static uint32_t tile[8];
  uint64_t pattern = 0;
  for (int i = 0; i < 8; ++i)
    tile[i] = 0x7F800000u; // +inf
  CHECK(nodata_find_nan(&pattern, tile, 8, 4) == 0);

  tile[3] = ODD_NAN;
  CHECK(nodata_find_nan(&pattern, tile, 8, 4) == 1);
  CHECK(pattern == ODD_NAN);
}

// The two shapes that used to skip the fill. There is one shape now, so both
// have to come back through it, and the all hole tile is the one that proves
// the wire needs no case of its own.
static void test_degenerate_tiles_round_trip(void) {
  static uint32_t tile[16], vals[16], out[16];
  static uint8_t m[16];

  for (int i = 0; i < 16; ++i)
    tile[i] = 0x40000000u + (uint32_t)i;
  nodata_mark_value(m, tile, 16, 4, 0xDEADBEEFu);
  for (int i = 0; i < 16; ++i)
    CHECK(m[i] == GEOZL_NODATA_VALID);
  nodata_fill(vals, tile, m, 4, 16, 4);
  nodata_restore(out, vals, m, 16, 4, 0xDEADBEEFu);
  CHECK(memcmp(out, tile, sizeof tile) == 0);

  for (int i = 0; i < 16; ++i)
    tile[i] = ODD_NAN;
  nodata_mark_nan(m, tile, 16, 4);
  for (int i = 0; i < 16; ++i)
    CHECK(m[i] == GEOZL_NODATA_INVALID);
  nodata_fill(vals, tile, m, 4, 16, 4);
  // Nothing measured to copy, so the fill is all zero and the mask carries the
  // whole tile.
  for (int i = 0; i < 16; ++i)
    CHECK(vals[i] == 0);
  nodata_restore(out, vals, m, 16, 4, ODD_NAN);
  CHECK(memcmp(out, tile, sizeof tile) == 0);
}

static void test_sentinel_all_widths(void) {
  static uint64_t tile[64], out[64], vals[64];
  static uint8_t m[64 * 8]; // one byte per sample, and a byte-wide tile has 512
  static const size_t widths[] = { 1, 2, 4, 8 };

  for (size_t w = 0; w < 4; ++w) {
    const size_t ew = widths[w];
    const uint64_t sentinel = (ew == 8) ? 0xFFFFFFFFFFFFFFFFull
                                        : (((uint64_t)1 << (8 * ew)) - 1);
    const size_t n = 64 * 8 / ew;
    memset(tile, 0x11, sizeof tile);
    // Scatter the sentinel so both fill branches run.
    for (size_t i = 0; i < n; i += 5)
      memcpy((uint8_t *)tile + i * ew, &sentinel, ew);

    nodata_mark_value(m, tile, n, ew, sentinel);
    for (size_t i = 0; i < n; ++i)
      CHECK(m[i] == ((i % 5 == 0) ? GEOZL_NODATA_INVALID : GEOZL_NODATA_VALID));
    nodata_fill(vals, tile, m, 16, n, ew);
    nodata_restore(out, vals, m, n, ew, sentinel);
    CHECK(memcmp(out, tile, n * ew) == 0);
  }
}

// The reason the codec exists. A raw sentinel drags the predictor across every
// hole edge, a filled hole does not.
static void test_fill_beats_sentinel(void) {
  static uint32_t sentinel_tile[N], res_raw[N], res_filled[N];
  uint64_t pattern = 0;

  nodata_find_nan(&pattern, src32, N, 4);
  nodata_mark_value(mask, src32, N, 4, pattern);
  nodata_fill(filled32, src32, mask, COLS, N, 4);
  memcpy(sentinel_tile, src32, sizeof src32);

  CHECK(planar_encode(res_raw, sentinel_tile, COLS, N, 4) == 0);
  CHECK(planar_encode(res_filled, filled32, COLS, N, 4) == 0);

  size_t nz_raw = 0, nz_filled = 0;
  for (size_t i = 0; i < N; ++i) {
    nz_raw += (res_raw[i] != 0);
    nz_filled += (res_filled[i] != 0);
  }
  if (!(nz_filled < nz_raw))
    printf("  FAIL fill did not flatten the holes, %zu vs %zu non-zero\n",
           nz_filled, nz_raw);
  CHECK(nz_filled < nz_raw);
}

// Every NaN is a hole whatever its payload. An infinity is not.
static void test_mark_nan_takes_every_payload(void) {
  static uint32_t tile[8];
  static uint8_t m[8];
  for (int i = 0; i < 8; ++i)
    tile[i] = 0x40000000u + (uint32_t)i;
  tile[1] = ODD_NAN;
  tile[5] = 0x7FA00001u; // a different payload
  tile[6] = 0x7F800000u; // +inf

  nodata_mark_nan(m, tile, 8, 4);
  for (int i = 0; i < 8; ++i)
    CHECK(m[i] == ((i == 1 || i == 5) ? GEOZL_NODATA_INVALID
                                      : GEOZL_NODATA_VALID));
}

// The kernels are reachable without a binding to reject the width first.
static void test_unsupported_width_leaves_a_readable_mask(void) {
  static uint8_t m[16], tile[48];
  memset(tile, 0, sizeof tile);

  memset(m, 0xAA, sizeof m);
  nodata_mark_nan(m, tile, 16, 3);
  for (int i = 0; i < 16; ++i)
    CHECK(m[i] == GEOZL_NODATA_VALID);

  memset(m, 0xAA, sizeof m);
  nodata_mark_value(m, tile, 16, 3, 0);
  for (int i = 0; i < 16; ++i)
    CHECK(m[i] == GEOZL_NODATA_VALID);
}

// Guarded sentinel tests.

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rnd(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static uint64_t all_bits(size_t w) {
  return (w == 8) ? ~(uint64_t)0 : (((uint64_t)1 << (8 * w)) - 1);
}

static void put(void *p, size_t w, uint64_t v) {
  switch (w) {
  case 1: *(uint8_t *)p = (uint8_t)v; break;
  case 2: *(uint16_t *)p = (uint16_t)v; break;
  case 4: *(uint32_t *)p = (uint32_t)v; break;
  default: *(uint64_t *)p = v; break;
  }
}

static uint64_t get(const void *p, size_t w) {
  switch (w) {
  case 1: return *(const uint8_t *)p;
  case 2: return *(const uint16_t *)p;
  case 4: return *(const uint32_t *)p;
  default: return *(const uint64_t *)p;
  }
}

static int is_float_dt(int dtype) { return dtype >= GEOZL_DT_F16; }

// Every f16 and f32 value is exact in a double.
static double as_real(int dtype, uint64_t b) {
  if (dtype == GEOZL_DT_F16)
    return (double)geozl_half_to_float((uint16_t)b);
  if (dtype == GEOZL_DT_F32) {
    const uint32_t u = (uint32_t)b;
    float f;
    memcpy(&f, &u, sizeof f);
    return (double)f;
  }
  double d;
  memcpy(&d, &b, sizeof d);
  return d;
}

// Compare bit patterns as values of dtype. Callers exclude NaN.
static int cmp_as(int dtype, uint64_t a, uint64_t b) {
  const size_t w = geozl_dtype_width(dtype);
  const uint64_t all = all_bits(w);
  if (is_float_dt(dtype)) {
    const double x = as_real(dtype, a), y = as_real(dtype, b);
    return (x > y) - (x < y);
  }
  a &= all;
  b &= all;
  if (dtype >= GEOZL_DT_I8) {
    const uint64_t sign = (uint64_t)1 << (8 * w - 1);
    a ^= sign;
    b ^= sign;
  }
  return (a > b) - (a < b);
}

static int is_nan_as(int dtype, uint64_t b) {
  return is_float_dt(dtype) && isnan(as_real(dtype, b));
}

// Returns the failing source line, or zero on success.
static int guard_case(int dtype, uint64_t s, uint64_t x) {
  const size_t w = geozl_dtype_width(dtype);
  const uint64_t all = all_bits(w);
  s &= all;
  x &= all;
  uint64_t repl[3];
  if (nodata_guard_values(repl, dtype, s) != 0)
    return __LINE__;
  uint64_t xin = 0, hit = 0, out = 0;
  uint8_t m = 0xAA;
  put(&xin, w, x);
  if (nodata_mark_guarded(&m, &xin, 1, dtype, s, INFINITY) != 0)
    return __LINE__;
  if (x == s)
    return m == GEOZL_NODATA_INVALID ? 0 : __LINE__;
  if (m < GEOZL_NODATA_ABOVE || m > GEOZL_NODATA_OTHER_ZERO)
    return __LINE__;

  if (nodata_restore_guarded(&out, &xin, &m, 1, w, s, repl) != 0)
    return __LINE__;
  if (get(&out, w) != x)
    return __LINE__;

  put(&hit, w, s);
  if (nodata_restore_guarded(&out, &hit, &m, 1, w, s, repl) != 0)
    return __LINE__;
  const uint64_t r = get(&out, w);
  if (r == s || r != repl[m - 1])
    return __LINE__;

  if (is_nan_as(dtype, x)) {
    const uint8_t want =
        (repl[0] != s) ? GEOZL_NODATA_ABOVE : GEOZL_NODATA_BELOW;
    return m == want ? 0 : __LINE__;
  }
  if (m == GEOZL_NODATA_OTHER_ZERO)
    return (r == x) ? 0 : __LINE__; // the other zero is the sample itself
  if (m == GEOZL_NODATA_ABOVE)
    return (cmp_as(dtype, x, s) > 0 && cmp_as(dtype, r, s) > 0 &&
            cmp_as(dtype, r, x) <= 0)
               ? 0
               : __LINE__;
  return (cmp_as(dtype, x, s) < 0 && cmp_as(dtype, r, s) < 0 &&
          cmp_as(dtype, r, x) >= 0)
             ? 0
             : __LINE__;
}

// Type boundaries, zero, infinities and normal/subnormal boundaries.
static size_t special_values(int dtype, uint64_t *out) {
  const size_t w = geozl_dtype_width(dtype);
  const uint64_t all = all_bits(w), sign = (uint64_t)1 << (8 * w - 1);
  size_t n = 0;
  out[n++] = 0;
  out[n++] = 1;
  out[n++] = 2;
  out[n++] = all;
  out[n++] = all - 1;
  out[n++] = sign;
  out[n++] = sign - 1;
  out[n++] = sign + 1;
  if (is_float_dt(dtype)) {
    uint64_t exp, one;
    if (dtype == GEOZL_DT_F16) {
      exp = 0x7C00u;
      one = 0x3C00u;
    } else if (dtype == GEOZL_DT_F32) {
      exp = 0x7F800000u;
      one = 0x3F800000u;
    } else {
      exp = 0x7FF0000000000000ull;
      one = 0x3FF0000000000000ull;
    }
    const uint64_t min_normal = exp & (~exp + 1); // lowest exponent bit
    out[n++] = exp;                               // +inf
    out[n++] = sign | exp;                        // -inf
    out[n++] = exp - 1;                           // largest finite
    out[n++] = sign | (exp - 1);                  // its negative
    out[n++] = min_normal;                        // smallest normal
    out[n++] = sign | min_normal;
    out[n++] = min_normal - 1;                    // largest subnormal
    out[n++] = one;
    out[n++] = sign | one;
    out[n++] = exp | 1;                           // a signalling NaN
    out[n++] = sign | exp | 1;
    out[n++] = exp | ((exp >> 1) & ~exp);         // the quiet NaN
  }
  return n;
}

static void guard_sweep(int dtype, const uint64_t *sents, size_t ns,
                        int all_samples, size_t random_samples) {
  const size_t w = geozl_dtype_width(dtype);
  const uint64_t all = all_bits(w);
  uint64_t specials[32];
  const size_t nsp = special_values(dtype, specials);
  long cases = 0;
  int first = 0;
  for (size_t i = 0; i < ns; ++i) {
    const uint64_t s = sents[i] & all;
    uint64_t repl[3];
    if (nodata_guard_values(repl, dtype, s) == 2)
      continue; // a NaN sentinel takes the plain form
#define TRY(x)                                                                 \
  do {                                                                         \
    const uint64_t xv = (x) & all;                                             \
    const int line = guard_case(dtype, s, xv);                                 \
    ++cases;                                                                   \
    if (line && !first) {                                                      \
      first = line;                                                            \
      printf("  FAIL dtype %d sentinel 0x%llx sample 0x%llx (check at line "   \
             "%d)\n",                                                          \
             dtype, (unsigned long long)s, (unsigned long long)xv, line);     \
    }                                                                          \
  } while (0)
    if (all_samples) {
      for (uint64_t x = 0; x <= all; ++x)
        TRY(x);
    } else {
      for (size_t k = 0; k < nsp; ++k)
        TRY(specials[k]);
      for (uint64_t d = 1; d <= 3; ++d) {
        TRY(s + d);
        TRY(s - d);
      }
      for (size_t k = 0; k < random_samples; ++k)
        TRY(rnd());
    }
#undef TRY
  }
  CHECK(first == 0);
  CHECK(cases > 0);
}

static size_t sentinel_set(int dtype, uint64_t *out, size_t random_count) {
  size_t n = special_values(dtype, out);
  for (size_t k = 0; k < random_count; ++k)
    out[n++] = rnd();
  return n;
}

static void test_guard_holds_the_bound(void) {
  static const int narrow[] = {GEOZL_DT_U8, GEOZL_DT_I8};
  static const int mid[] = {GEOZL_DT_U16, GEOZL_DT_I16, GEOZL_DT_F16};
  static const int wide[] = {GEOZL_DT_U32, GEOZL_DT_I32, GEOZL_DT_F32,
                             GEOZL_DT_U64, GEOZL_DT_I64, GEOZL_DT_F64};
  uint64_t sents[256];

  for (size_t t = 0; t < 2; ++t) {
    for (uint64_t s = 0; s < 256; ++s)
      sents[s] = s;
    guard_sweep(narrow[t], sents, 256, 1, 0);
  }
  for (size_t t = 0; t < 3; ++t)
    guard_sweep(mid[t], sents, sentinel_set(mid[t], sents, 24), 1, 0);
  for (size_t t = 0; t < 6; ++t)
    guard_sweep(wide[t], sents, sentinel_set(wide[t], sents, 48), 0, 4096);
}

// Check float neighbours against libm.
static void test_guard_values_match_nextafter(void) {
  uint64_t sents[128];
  const size_t n32 = sentinel_set(GEOZL_DT_F32, sents, 64);
  for (size_t i = 0; i < n32; ++i) {
    const uint32_t s = (uint32_t)sents[i];
    float f;
    memcpy(&f, &s, sizeof f);
    uint64_t repl[3];
    const int rc = nodata_guard_values(repl, GEOZL_DT_F32, s);
    if (isnan(f)) {
      CHECK(rc == 2);
      continue;
    }
    CHECK(rc == 0);
    const float up = nextafterf(f, INFINITY), down = nextafterf(f, -INFINITY);
    uint32_t bu, bd;
    memcpy(&bu, &up, sizeof bu);
    memcpy(&bd, &down, sizeof bd);
    CHECK(repl[0] == bu);
    CHECK(repl[1] == bd);
    CHECK(repl[2] == ((f == 0.0f) ? (uint64_t)(s ^ 0x80000000u) : s));
  }
  const size_t n64 = sentinel_set(GEOZL_DT_F64, sents, 64);
  for (size_t i = 0; i < n64; ++i) {
    const uint64_t s = sents[i];
    double d;
    memcpy(&d, &s, sizeof d);
    uint64_t repl[3];
    const int rc = nodata_guard_values(repl, GEOZL_DT_F64, s);
    if (isnan(d)) {
      CHECK(rc == 2);
      continue;
    }
    CHECK(rc == 0);
    const double up = nextafter(d, INFINITY), down = nextafter(d, -INFINITY);
    uint64_t bu, bd;
    memcpy(&bu, &up, sizeof bu);
    memcpy(&bd, &down, sizeof bd);
    CHECK(repl[0] == bu);
    CHECK(repl[1] == bd);
    CHECK(repl[2] == ((d == 0.0) ? (s ^ 0x8000000000000000ull) : s));
  }
}

// float16 has no nextafter, so verify adjacency exhaustively.
static void test_guard_values_are_adjacent_in_half(void) {
  uint64_t sents[64];
  const size_t n = sentinel_set(GEOZL_DT_F16, sents, 24);
  for (size_t i = 0; i < n; ++i) {
    const uint16_t s = (uint16_t)sents[i];
    uint64_t repl[3];
    const int rc = nodata_guard_values(repl, GEOZL_DT_F16, s);
    const double fs = as_real(GEOZL_DT_F16, s);
    if (isnan(fs)) {
      CHECK(rc == 2);
      continue;
    }
    CHECK(rc == 0);
    const double up = as_real(GEOZL_DT_F16, repl[0]);
    const double down = as_real(GEOZL_DT_F16, repl[1]);
    CHECK(repl[0] == s ? isinf(fs) && fs > 0 : up > fs);
    CHECK(repl[1] == s ? isinf(fs) && fs < 0 : down < fs);
    for (uint32_t h = 0; h < 65536; ++h) {
      const double v = as_real(GEOZL_DT_F16, h);
      if (isnan(v))
        continue;
      if (repl[0] != s && v > fs && v < up) {
        CHECK(0);
        break;
      }
      if (repl[1] != s && v < fs && v > down) {
        CHECK(0);
        break;
      }
    }
  }
}

static void test_guard_values_at_the_integer_ends(void) {
  uint64_t r[3];
  CHECK(nodata_guard_values(r, GEOZL_DT_U8, 0) == 0);
  CHECK(r[0] == 1 && r[1] == 0 && r[2] == 0);
  CHECK(nodata_guard_values(r, GEOZL_DT_U8, 255) == 0);
  CHECK(r[0] == 255 && r[1] == 254 && r[2] == 255);
  CHECK(nodata_guard_values(r, GEOZL_DT_I8, 0x80) == 0); // -128
  CHECK(r[0] == 0x81 && r[1] == 0x80);
  CHECK(nodata_guard_values(r, GEOZL_DT_I8, 0x7F) == 0); // 127
  CHECK(r[0] == 0x7F && r[1] == 0x7E);
  CHECK(nodata_guard_values(r, GEOZL_DT_I16, 0xFFFF) == 0); // -1 steps to 0
  CHECK(r[0] == 0 && r[1] == 0xFFFE);
  CHECK(nodata_guard_values(r, GEOZL_DT_U64, ~(uint64_t)0) == 0);
  CHECK(r[0] == ~(uint64_t)0 && r[1] == ~(uint64_t)1);
  CHECK(nodata_guard_values(r, GEOZL_DT_I64, 0x8000000000000000ull) == 0);
  CHECK(r[0] == 0x8000000000000001ull && r[1] == 0x8000000000000000ull);
  // Bits above the element are not part of the sentinel.
  CHECK(nodata_guard_values(r, GEOZL_DT_U16, 0xABCD0005ull) == 0);
  CHECK(r[0] == 6 && r[1] == 4 && r[2] == 5);
}

static void test_guard_refusals(void) {
  uint64_t r[3];
  uint8_t m[4];
  const uint32_t tile[4] = {0x7FC00000u, 0x3F800000u, 0x7FC00000u, 0};

  CHECK(nodata_guard_values(r, 11, 0) == 1);
  CHECK(nodata_guard_values(r, -1, 0) == 1);
  memset(m, 0xAA, sizeof m);
  CHECK(nodata_mark_guarded(m, tile, 4, 11, 0, INFINITY) == 1);
  for (int i = 0; i < 4; ++i)
    CHECK(m[i] == GEOZL_NODATA_VALID);

  // A NaN sentinel: the plain marking, holes on its exact bits only.
  CHECK(nodata_guard_values(r, GEOZL_DT_F32, 0x7FC00000u) == 2);
  memset(m, 0xAA, sizeof m);
  CHECK(nodata_mark_guarded(m, tile, 4, GEOZL_DT_F32, 0x7FC00000u, INFINITY) == 2);
  CHECK(m[0] == GEOZL_NODATA_INVALID && m[1] == GEOZL_NODATA_VALID);
  CHECK(m[2] == GEOZL_NODATA_INVALID && m[3] == GEOZL_NODATA_VALID);
  CHECK(nodata_guard_values(r, GEOZL_DT_F16, 0x7E00u) == 2);
}

// The decoder rejects mask states the encoder cannot produce.
static void test_restore_guarded_refuses_forged_masks(void) {
  uint64_t r[3];
  uint8_t vals[4] = {10, 200, 30, 40}, out[4], m[4];

  CHECK(nodata_guard_values(r, GEOZL_DT_U8, 200) == 0);
  m[0] = GEOZL_NODATA_BELOW;
  m[1] = GEOZL_NODATA_INVALID;
  m[2] = GEOZL_NODATA_BELOW;
  m[3] = GEOZL_NODATA_BELOW;
  CHECK(nodata_restore_guarded(out, vals, m, 4, 1, 200, r) == 0);
  CHECK(out[0] == 10 && out[1] == 200 && out[2] == 30);

  m[3] = 4;
  memset(out, 0xEE, sizeof out);
  CHECK(nodata_restore_guarded(out, vals, m, 4, 1, 200, r) != 0);
  CHECK(out[0] == 10 && out[1] == 200 && out[2] == 30);

  m[3] = GEOZL_NODATA_VALID; // plain form code, never in the guarded form
  CHECK(nodata_restore_guarded(out, vals, m, 4, 1, 200, r) != 0);

  CHECK(nodata_guard_values(r, GEOZL_DT_U8, 255) == 0);
  m[0] = GEOZL_NODATA_BELOW;
  m[1] = GEOZL_NODATA_ABOVE;
  m[2] = m[3] = GEOZL_NODATA_INVALID;
  CHECK(nodata_restore_guarded(out, vals, m, 4, 1, 255, r) != 0);
  CHECK(nodata_guard_values(r, GEOZL_DT_U8, 0) == 0);
  m[0] = GEOZL_NODATA_ABOVE;
  m[1] = GEOZL_NODATA_OTHER_ZERO;
  CHECK(nodata_restore_guarded(out, vals, m, 4, 1, 0, r) != 0);

  CHECK(nodata_restore_guarded(out, vals, m, 1, 3, 0, r) != 0);
}

// Guarded encoding remains bit-exact without a lossy stage.
static void test_guarded_round_trip_every_dtype(void) {
  static uint64_t tile[512], vals[512], out[512];
  static uint8_t m[512];
  for (int dt = GEOZL_DT_U8; dt <= GEOZL_DT_F64; ++dt) {
    const size_t w = geozl_dtype_width(dt);
    uint64_t sents[64];
    const size_t ns = sentinel_set(dt, sents, 8);
    for (size_t k = 0; k < ns; ++k) {
      const uint64_t s = sents[k] & all_bits(w);
      const size_t n = 300;
      for (size_t i = 0; i < n; ++i)
        put((uint8_t *)tile + i * w, w, (i % 7 == 0) ? s : rnd());
      const int rc = nodata_mark_guarded(m, tile, n, dt, s, INFINITY);
      uint64_t r[3];
      nodata_guard_values(r, dt, s);
      nodata_fill(vals, tile, m, 20, n, w);
      if (rc == 2) {
        nodata_restore(out, vals, m, n, w, s);
      } else {
        CHECK(rc == 0);
        for (size_t i = 0; i < n; ++i)
          CHECK(m[i] <= GEOZL_NODATA_OTHER_ZERO);
        CHECK(nodata_restore_guarded(out, vals, m, n, w, s, r) == 0);
      }
      CHECK(memcmp(out, tile, n * w) == 0);
    }
  }
}

// Samples outside radius use the default side.
static int radius_case(int dtype, uint64_t s, uint64_t x, double radius) {
  const size_t w = geozl_dtype_width(dtype);
  const uint64_t all = all_bits(w);
  uint64_t repl[3], xin = 0, hit = 0, out = 0;
  uint8_t m = 0xAA;
  if (nodata_guard_values(repl, dtype, s) != 0)
    return __LINE__;
  put(&xin, w, x);
  if (nodata_mark_guarded(&m, &xin, 1, dtype, s, radius) != 0)
    return __LINE__;
  if (x == s)
    return m == GEOZL_NODATA_INVALID ? 0 : __LINE__;
  const uint8_t def =
      (repl[0] != s) ? GEOZL_NODATA_ABOVE : GEOZL_NODATA_BELOW;
  put(&hit, w, s);
  if (nodata_restore_guarded(&out, &hit, &m, 1, w, s, repl) != 0)
    return __LINE__;
  const uint64_t r = get(&out, w) & all;
  if (r == s)
    return __LINE__;
  // Distance in whole units for an integer, the only types swept here.
  const uint64_t sign = (uint64_t)1 << (8 * w - 1);
  const uint64_t kx = (dtype >= GEOZL_DT_I8) ? (x ^ sign) : x;
  const uint64_t ks = (dtype >= GEOZL_DT_I8) ? (s ^ sign) : s;
  const uint64_t d = kx > ks ? kx - ks : ks - kx;
  if ((double)d <= radius) {
    const int want = kx > ks ? GEOZL_NODATA_ABOVE : GEOZL_NODATA_BELOW;
    if (m != want)
      return __LINE__;
    const uint64_t kr = (dtype >= GEOZL_DT_I8) ? (r ^ sign) : r;
    const uint64_t dr = kx > kr ? kx - kr : kr - kx;
    return dr < d ? 0 : __LINE__;
  }
  return m == def ? 0 : __LINE__;
}

static void test_guard_radius_every_8_bit_case(void) {
  static const double radii[] = {0.0, 1.0, 5.0, 40.0, 254.0};
  static const int types[] = {GEOZL_DT_U8, GEOZL_DT_I8};
  int first = 0;
  for (size_t t = 0; t < 2; ++t)
    for (size_t k = 0; k < 5; ++k)
      for (uint64_t s = 0; s < 256; ++s)
        for (uint64_t x = 0; x < 256; ++x) {
          const int line = radius_case(types[t], s, x, radii[k]);
          if (line && !first) {
            first = line;
            printf("  FAIL dtype %d radius %g sentinel %llu sample %llu (line "
                   "%d)\n",
                   types[t], radii[k], (unsigned long long)s,
                   (unsigned long long)x, line);
          }
        }
  CHECK(first == 0);
}

static void test_guard_radius_codes(void) {
  uint8_t m[6];
  // i16, sentinel 0 inside the data, radius 3: sides near, default far.
  const int16_t v[6] = {2, -2, 3, -3, 10, -10};
  CHECK(nodata_mark_guarded(m, v, 6, GEOZL_DT_I16, 0, 3.0) == 0);
  CHECK(m[0] == GEOZL_NODATA_ABOVE && m[1] == GEOZL_NODATA_BELOW);
  CHECK(m[2] == GEOZL_NODATA_ABOVE && m[3] == GEOZL_NODATA_BELOW);
  CHECK(m[4] == GEOZL_NODATA_ABOVE && m[5] == GEOZL_NODATA_ABOVE);
  // u16 at the top: the default is the only side there is.
  const uint16_t u[3] = {65534, 100, 65535};
  CHECK(nodata_mark_guarded(m, u, 3, GEOZL_DT_U16, 65535, 0.0) == 0);
  CHECK(m[0] == GEOZL_NODATA_BELOW && m[1] == GEOZL_NODATA_BELOW);
  CHECK(m[2] == GEOZL_NODATA_INVALID);
  // f32, zero sentinel, radius 0: the other zero still gets its code, every
  // other sample, a NaN included, the default.
  const uint32_t f[5] = {0x80000000u, 0x3F800000u, 0xBF800000u, 0x7FC00000u,
                         0x00000000u};
  CHECK(nodata_mark_guarded(m, f, 5, GEOZL_DT_F32, 0, 0.0) == 0);
  CHECK(m[0] == GEOZL_NODATA_OTHER_ZERO && m[1] == GEOZL_NODATA_ABOVE);
  CHECK(m[2] == GEOZL_NODATA_ABOVE && m[3] == GEOZL_NODATA_ABOVE);
  CHECK(m[4] == GEOZL_NODATA_INVALID);
  // radius 2 reaches -1: its own side.
  CHECK(nodata_mark_guarded(m, f, 5, GEOZL_DT_F32, 0, 2.0) == 0);
  CHECK(m[2] == GEOZL_NODATA_BELOW);
  // An unknown radius, NaN or negative, records every side.
  CHECK(nodata_mark_guarded(m, v, 6, GEOZL_DT_I16, 0, -1.0) == 0);
  CHECK(m[5] == GEOZL_NODATA_BELOW);
  CHECK(nodata_mark_guarded(m, v, 6, GEOZL_DT_I16, 0, NAN) == 0);
  CHECK(m[5] == GEOZL_NODATA_BELOW);
}

// Restore changes only collisions and supports in-place decoding.
static void test_restore_guarded_fixes_only_collisions(void) {
  uint64_t r[3];
  uint16_t vals[6] = {100, 7, 100, 50, 100, 9}, out[6];
  const uint8_t m[6] = {GEOZL_NODATA_ABOVE, GEOZL_NODATA_INVALID,
                        GEOZL_NODATA_BELOW, GEOZL_NODATA_ABOVE,
                        GEOZL_NODATA_INVALID, GEOZL_NODATA_BELOW};
  CHECK(nodata_guard_values(r, GEOZL_DT_U16, 100) == 0);
  CHECK(nodata_restore_guarded(out, vals, m, 6, 2, 100, r) == 0);
  CHECK(out[0] == 101 && out[1] == 100 && out[2] == 99);
  CHECK(out[3] == 50 && out[4] == 100 && out[5] == 9);
  CHECK(nodata_restore_guarded(vals, vals, m, 6, 2, 100, r) == 0);
  CHECK(memcmp(vals, out, sizeof out) == 0);
}

// Compare each recipe radius directly with its mathematical bound.
static double bound_at(const geozl_lossy_recipe *r, double x) {
  switch (r->family) {
  case GEOZL_LOSSY_LINEAR:
    return r->as.linear.max_error;
  case GEOZL_LOSSY_LOG:
    return r->as.log.rel_err * fabs(x);
  case GEOZL_LOSSY_SQRT: {
    const double v = r->as.sqrt.a + r->as.sqrt.b * x;
    return v >= 0.0 ? r->as.sqrt.k * sqrt(v) : -1.0; // outside the domain
  }
  default:
    return 0.0;
  }
}

static void test_guard_radius_reaches_the_bound(void) {
  static const char *recipes[] = {
      "LINEAR:MAX_ERROR=0.5", "LINEAR:MAX_ERROR=3", "LINEAR:MAX_ERROR=100",
      "LOG:MAX_ERROR=1%",     "LOG:MAX_ERROR=5%",   "LOG:MAX_ERROR=40%",
      "SQRT:MAX_ERROR=1N,A=1,B=1", "SQRT:MAX_ERROR=2N,A=11,B=1",
      "SQRT:MAX_ERROR=0.5N,A=0,B=3",
  };
  static const double sents[] = {-1000.0, -1.0, 0.0, 1.0, 100.0, 1e6};
  char err[160];
  for (size_t i = 0; i < sizeof recipes / sizeof recipes[0]; ++i) {
    geozl_lossy_recipe r;
    CHECK(geozl_lossy_parse(recipes[i], &r, err, sizeof err) == 0);
    for (size_t j = 0; j < sizeof sents / sizeof sents[0]; ++j) {
      const double s = sents[j];
      const double radius = geozl_lossy_guard_radius(&r, s);
      CHECK(radius >= 0.0 && isfinite(radius));
      const double span = 4.0 * radius + 16.0;
      const int steps = 400000;
      double reach = 0.0;
      int short_at = 0;
      for (int k = -steps; k <= steps; ++k) {
        const double x = s + span * (double)k / (double)steps;
        const double b = bound_at(&r, x);
        if (b >= 0.0 && fabs(x - s) <= b) {
          reach = fmax(reach, fabs(x - s));
          if (fabs(x - s) > radius && !short_at) {
            short_at = 1;
            printf("  FAIL %s sentinel %g: x = %.9g reaches %.9g past radius "
                   "%.9g\n",
                   recipes[i], s, x, fabs(x - s), radius);
          }
        }
      }
      CHECK(!short_at);
      if (!(radius <= reach + 2.0 * span / (double)steps + 1e-9 * radius))
        printf("  LOOSE %s sentinel %g: radius %.9g reach %.9g\n", recipes[i],
               s, radius, reach);
      CHECK(radius <= reach + 2.0 * span / (double)steps + 1e-9 * radius);
    }
  }
  geozl_lossy_recipe none;
  CHECK(geozl_lossy_parse(NULL, &none, err, sizeof err) == 0);
  CHECK(geozl_lossy_guard_radius(&none, 5.0) == 0.0);
}

int main(void) {
  build_tile();
  test_nan_round_trip();
  test_infinity_is_a_value();
  test_mark_nan_takes_every_payload();
  test_unsupported_width_leaves_a_readable_mask();
  test_degenerate_tiles_round_trip();
  test_sentinel_all_widths();
  test_fill_beats_sentinel();
  test_guard_values_at_the_integer_ends();
  test_guard_values_match_nextafter();
  test_guard_values_are_adjacent_in_half();
  test_guard_refusals();
  test_restore_guarded_refuses_forged_masks();
  test_guarded_round_trip_every_dtype();
  test_guard_holds_the_bound();
  test_guard_radius_codes();
  test_guard_radius_every_8_bit_case();
  test_restore_guarded_fixes_only_collisions();
  test_guard_radius_reaches_the_bound();

  if (failures) {
    printf("test_nodata: %d failed\n", failures);
    return 1;
  }
  printf("test_nodata: ok\n");
  return 0;
}
