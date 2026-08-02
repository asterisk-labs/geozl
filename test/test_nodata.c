#include "nodata/decode_nodata_kernel.h"
#include "nodata/encode_nodata_kernel.h"
#include "planar/encode_planar_kernel.h"

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

  const size_t marked = nodata_mark_value(mask, src32, N, 4, pattern);
  CHECK(marked > 0 && marked < N);

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

// The two shapes that carry no mask. Neither goes through nodata_restore, so
// what the kernels owe is a count of 0 and a count of nbElts.
static void test_all_valid_and_all_hole(void) {
  static uint32_t tile[16], out[16];
  static uint8_t m[16];

  for (int i = 0; i < 16; ++i)
    tile[i] = 0x40000000u + (uint32_t)i;
  CHECK(nodata_mark_nan(m, tile, 16, 4) == 0);
  CHECK(nodata_mark_value(m, tile, 16, 4, 0xDEADBEEFu) == 0);

  for (int i = 0; i < 16; ++i)
    tile[i] = ODD_NAN;
  CHECK(nodata_mark_nan(m, tile, 16, 4) == 16);
  nodata_broadcast(out, 16, 4, ODD_NAN);
  CHECK(memcmp(out, tile, sizeof tile) == 0);

  // Every width, since the broadcast switches on it.
  static uint64_t wide[8], wout[8];
  for (int i = 0; i < 8; ++i)
    wide[i] = 0x0102030405060708ull;
  nodata_broadcast(wout, 8, 8, 0x0102030405060708ull);
  CHECK(memcmp(wout, wide, sizeof wide) == 0);
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

    const size_t hits = nodata_mark_value(m, tile, n, ew, sentinel);
    CHECK(hits == (n + 4) / 5);
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

int main(void) {
  build_tile();
  test_nan_round_trip();
  test_infinity_is_a_value();
  test_all_valid_and_all_hole();
  test_sentinel_all_widths();
  test_fill_beats_sentinel();

  if (failures) {
    printf("test_nodata: %d failed\n", failures);
    return 1;
  }
  printf("test_nodata: ok\n");
  return 0;
}
