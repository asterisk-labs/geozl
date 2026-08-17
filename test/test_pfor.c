#include "pfor/decode_pfor_kernel.h"
#include "pfor/encode_pfor_kernel.h"
#include "pfor/pfor_check.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                 \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define MAXN 5000

// One aligned buffer for every element width.
static uint64_t src[MAXN], back[MAXN];
static uint8_t packed[MAXN * 8 + 4096];

static uint64_t rs = 0x243F6A8885A308D3ull;
static uint64_t rnd(void) {
  rs ^= rs << 13;
  rs ^= rs >> 7;
  rs ^= rs << 17;
  return rs;
}

static void put(void *b, size_t i, size_t w, uint64_t v) {
  switch (w) {
  case 1: ((uint8_t *)b)[i] = (uint8_t)v; break;
  case 2: ((uint16_t *)b)[i] = (uint16_t)v; break;
  case 4: ((uint32_t *)b)[i] = (uint32_t)v; break;
  default: ((uint64_t *)b)[i] = v; break;
  }
}

static void trip(const char *what, size_t n, size_t w) {
  const size_t bound = pfor_bound(n, w);
  size_t written = 0;
  if (bound > sizeof(packed)) {
    printf("  FAIL %s: bound %zu over the buffer\n", what, bound);
    ++failures;
    return;
  }
  if (pfor_encode(packed, bound, &written, src, n, w) != 0) {
    printf("  FAIL %s: encode refused, n=%zu w=%zu\n", what, n, w);
    ++failures;
    return;
  }
  CHECK(written <= bound);
  memset(back, 0xAB, n * w);
  if (pfor_decode(back, n, w, packed, written) != 0) {
    printf("  FAIL %s: decode refused, n=%zu w=%zu\n", what, n, w);
    ++failures;
    return;
  }
  if (memcmp(back, src, n * w) != 0) {
    printf("  FAIL %s: mismatch, n=%zu w=%zu\n", what, n, w);
    ++failures;
  }
}

static const size_t kW[4] = {1, 2, 4, 8};
static const size_t kN[] = {1, 2, 127, 128, 129, 255, 256, 257, 1000, 4096};

static void test_every_bit_width(void) {
  for (int wi = 0; wi < 4; ++wi) {
    const size_t w = kW[wi];
    for (unsigned bits = 0; bits <= 8 * w; ++bits) {
      const uint64_t lim = (bits >= 64) ? ~0ull : ((1ull << bits) - 1ull);
      for (size_t si = 0; si < sizeof(kN) / sizeof(kN[0]); ++si) {
        const size_t n = kN[si];
        for (size_t i = 0; i < n; ++i)
          put(src, i, w, lim ? (rnd() & lim) : 0);
        put(src, 0, w, lim);
        trip("bit width", n, w);
      }
    }
  }
}

static void test_exceptions(void) {
  const unsigned pct[] = {0, 1, 3, 12, 13, 50, 99, 100};
  for (int wi = 0; wi < 4; ++wi) {
    const size_t w = kW[wi];
    const uint64_t lo = (1ull << (4 * w)) - 1ull;
    const uint64_t hi = (8 * w >= 64) ? ~0ull : ((1ull << (8 * w)) - 1ull);
    for (size_t pi = 0; pi < sizeof(pct) / sizeof(pct[0]); ++pi)
      for (size_t si = 0; si < sizeof(kN) / sizeof(kN[0]); ++si) {
        const size_t n = kN[si];
        for (size_t i = 0; i < n; ++i)
          put(src, i, w, ((rnd() % 100u) < pct[pi]) ? (rnd() & hi) : (rnd() & lo));
        trip("exceptions", n, w);
      }
  }
}

static void test_degenerate_blocks(void) {
  for (int wi = 0; wi < 4; ++wi) {
    const size_t w = kW[wi];
    for (size_t si = 0; si < sizeof(kN) / sizeof(kN[0]); ++si) {
      const size_t n = kN[si];
      memset(src, 0x00, n * w); trip("all zero", n, w);
      memset(src, 0xFF, n * w); trip("all max", n, w);
      for (size_t i = 0; i < n; ++i)
        put(src, i, w, (i % 2) ? 0 : ~0ull);
      trip("alternating", n, w);
    }
  }
}

static void test_refusals(void) {
  size_t out = 0;
  memset(src, 0, sizeof(src));
  CHECK(pfor_encode(packed, sizeof(packed), &out, src, 0, 2) != 0);
  CHECK(pfor_encode(packed, sizeof(packed), &out, src, 4, 3) != 0);
  CHECK(pfor_encode(NULL, sizeof(packed), &out, src, 4, 2) != 0);
  CHECK(pfor_encode(packed, sizeof(packed), NULL, src, 4, 2) != 0);
  CHECK(pfor_encode(packed, 4, &out, src, GEOZL_PFOR_BLOCK, 2) != 0);
  CHECK(pfor_bound(0, 2) == 0);
  CHECK(pfor_bound(4, 3) == 0);
  CHECK(pfor_bound(SIZE_MAX, 8) == 0);
  CHECK(pfor_bound(SIZE_MAX / 2, 8) == 0);
  CHECK(pfor_decode(back, 4, 2, packed, 0) != 0);
  CHECK(pfor_decode(back, 4, 7, packed, 8) != 0);
  CHECK(pfor_decode(back, 0, 2, packed, 8) != 0);
  CHECK(pfor_decode(NULL, 4, 2, packed, 8) != 0);
}

static void test_truncation_and_flips(void) {
  for (int wi = 0; wi < 4; ++wi) {
    const size_t w = kW[wi];
    const size_t n = 512;
    for (size_t i = 0; i < n; ++i)
      put(src, i, w, ((rnd() % 100u) < 8u) ? rnd() : (rnd() & 0xF));
    size_t written = 0;
    if (pfor_encode(packed, pfor_bound(n, w), &written, src, n, w) != 0) {
      printf("  FAIL truncation setup, w=%zu\n", w);
      ++failures;
      continue;
    }
    for (size_t len = 0; len < written; ++len)
      CHECK(pfor_decode(back, n, w, packed, len) != 0);
    uint8_t *longer = (uint8_t *)malloc(written + 1);
    memcpy(longer, packed, written);
    longer[written] = 0;
    CHECK(pfor_decode(back, n, w, longer, written + 1) != 0);
    free(longer);

    uint8_t *m = (uint8_t *)malloc(written);
    for (size_t i = 0; i < written; ++i)
      for (int bit = 0; bit < 8; ++bit) {
        memcpy(m, packed, written);
        m[i] ^= (uint8_t)(1u << bit);
        (void)pfor_decode(back, n, w, m, written);
      }
    free(m);
  }
}

int main(void) {
  test_every_bit_width();
  test_exceptions();
  test_degenerate_blocks();
  test_refusals();
  test_truncation_and_flips();

  if (failures) {
    printf("test_pfor: %d failed\n", failures);
    return 1;
  }
  printf("test_pfor: ok\n");
  return 0;
}
