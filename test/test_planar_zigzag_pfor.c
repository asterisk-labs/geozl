#include "pfor/decode_pfor_kernel.h"
#include "pfor/encode_pfor_kernel.h"
#include "planar_zigzag_pfor/decode_planar_zigzag_pfor_kernel.h"
#include "planar_zigzag_pfor/encode_planar_zigzag_pfor_kernel.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);               \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define MAXN 40000
#define MAXOUT (MAXN * 8 * 3 + 65536)

static uint64_t source[MAXN], decoded[MAXN];
static uint8_t got[MAXOUT], want[MAXOUT];

static uint64_t rng = UINT64_C(0x243f6a8885a308d3);
static uint64_t rnd(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

static uint64_t mask_of(size_t elt) {
  return elt == 8 ? ~UINT64_C(0) : (UINT64_C(1) << (8 * elt)) - 1;
}

static uint64_t get(const void *p, size_t i, size_t elt) {
  switch (elt) {
  case 1:
    return ((const uint8_t *)p)[i];
  case 2:
    return ((const uint16_t *)p)[i];
  case 4:
    return ((const uint32_t *)p)[i];
  default:
    return ((const uint64_t *)p)[i];
  }
}

static void put(void *p, size_t i, size_t elt, uint64_t v) {
  switch (elt) {
  case 1:
    ((uint8_t *)p)[i] = (uint8_t)v;
    break;
  case 2:
    ((uint16_t *)p)[i] = (uint16_t)v;
    break;
  case 4:
    ((uint32_t *)p)[i] = (uint32_t)v;
    break;
  default:
    ((uint64_t *)p)[i] = v;
    break;
  }
}

// Independent layout reference using only the standalone PFOR block encoder.
#define REF_STRIP 1024

static unsigned ref_band(size_t rowsLeft) {
  if (rowsLeft >= 16)
    return 16;
  if (rowsLeft >= 8)
    return 8;
  if (rowsLeft >= 4)
    return 4;
  return rowsLeft >= 2 ? 2 : 1;
}

static size_t ref_blocks(size_t width, size_t rows, uint32_t planes) {
  size_t blocks = 0;
  for (size_t y = 0; y < rows;) {
    const unsigned h = ref_band(rows - y);
    const size_t bw = 256 / h;
    blocks += (width + bw - 1) / bw;
    y += h;
  }
  return blocks * planes;
}

static size_t ref_encode(uint8_t *dst, size_t width, size_t n, size_t elt,
                         uint32_t planes) {
  const uint64_t mask = mask_of(elt);
  const size_t planeElts = n / planes, rows = planeElts / width;
  uint8_t *hdr = dst;
  uint8_t *body = dst + 3 * ref_blocks(width, rows, planes);
  uint64_t z[256];
  uint8_t block[2 + 256 * 8];
  for (uint32_t plane = 0; plane < planes; ++plane) {
    const size_t base = plane * planeElts;
    for (size_t y0 = 0; y0 < rows;) {
      const unsigned h = ref_band(rows - y0);
      const size_t bw = 256 / h;
      for (size_t x0 = 0; x0 < width; x0 += bw) {
        memset(z, 0, sizeof(z));
        for (size_t i = 0; i < h; ++i)
          for (size_t j = 0; j < bw && x0 + j < width; ++j) {
            const size_t y = y0 + i, x = x0 + j;
            const int left = x % REF_STRIP != 0;
            const uint64_t s = get(source, base + y * width + x, elt);
            const uint64_t w =
                left ? get(source, base + y * width + x - 1, elt) : 0;
            const uint64_t nn =
                i ? get(source, base + (y - 1) * width + x, elt) : 0;
            const uint64_t nw =
                (i && left) ? get(source, base + (y - 1) * width + x - 1, elt)
                            : 0;
            const uint64_t r = (s - (w + nn - nw)) & mask;
            const uint64_t sign = ((r >> (8 * elt - 1)) & 1) ? mask : 0;
            put(z, 8 * (i + h * (j / 8)) + j % 8, elt,
                ((r << 1) ^ sign) & mask);
          }
        const size_t used = pfor_encode_block(block, z, elt);
        const size_t head = block[1] ? 3 : 2;
        hdr[0] = block[0];
        hdr[1] = block[1];
        hdr[2] = block[1] ? block[2] : 0;
        hdr += 3;
        memcpy(body, block + head, used - head);
        body += used - head;
      }
      y0 += h;
    }
  }
  return (size_t)(body - dst);
}

enum { RANDOM, SMOOTH, FLAT, FEW_OUTLIERS, MANY_OUTLIERS, PATTERNS };

static void fill(size_t n, size_t width, size_t elt, int pattern) {
  const uint64_t mask = mask_of(elt);
  for (size_t k = 0; k < n; ++k) {
    const size_t y = k / width, x = k % width;
    uint64_t v;
    switch (pattern) {
    case RANDOM:
      v = rnd();
      break;
    case FLAT:
      v = UINT64_C(0x5A5A5A5A5A5A5A5A);
      break;
    default:
      v = 1000 + 3 * x + 2 * y + (rnd() & 3);
      if (pattern == FEW_OUTLIERS && rnd() % 97 == 0)
        v += rnd();
      if (pattern == MANY_OUTLIERS && rnd() % 5 == 0)
        v += rnd();
      break;
    }
    put(source, k, elt, v & mask);
  }
}

static void trip(size_t width, size_t rows, uint32_t planes, size_t elt,
                 int pattern) {
  const size_t n = width * rows * planes;
  CHECK(n <= MAXN);
  if (n > MAXN)
    return;
  fill(n, width, elt, pattern);
  const size_t blocks = ref_blocks(width, rows, planes);
  const size_t bound = planar_zigzag_pfor_bound(n, elt, width, planes);
  CHECK(bound == blocks * (3 + 256 * elt));
  CHECK(bound <= MAXOUT);
  if (bound > MAXOUT)
    return;

  size_t gotSize = 0;
  CHECK(planar_zigzag_pfor_encode(got, bound, &gotSize, source, width, n, elt,
                                  planes) == 0);
  const size_t wantSize = ref_encode(want, width, n, elt, planes);
  if (gotSize != wantSize || memcmp(got, want, wantSize) != 0) {
    printf("  FAIL layout differs: width %zu rows %zu planes %u elt %zu "
           "pattern %d\n",
           width, rows, planes, elt, pattern);
    ++failures;
    return;
  }
  CHECK(gotSize <= bound);

  memset(decoded, 0xAB, n * elt);
  CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, planes, got,
                                  gotSize) == 0);
  CHECK(memcmp(decoded, source, n * elt) == 0);

  CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, planes, got,
                                  gotSize - 1) != 0);
  got[gotSize] = 0;
  CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, planes, got,
                                  gotSize + 1) != 0);
}

// Every truncation of a small payload fails.
static void truncations(size_t width, size_t rows, size_t elt) {
  const size_t n = width * rows;
  fill(n, width, elt, MANY_OUTLIERS);
  size_t size = 0;
  CHECK(planar_zigzag_pfor_encode(got, sizeof(got), &size, source, width, n,
                                  elt, 1) == 0);
  for (size_t len = 0; len < size; ++len) {
    uint8_t *copy = malloc(len ? len : 1);
    CHECK(copy != NULL);
    if (copy == NULL)
      return;
    memcpy(copy, got, len);
    CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, 1, copy, len) != 0);
    free(copy);
  }
}

static void forgeries(void) {
  const size_t width = 64, rows = 32, n = width * rows, elt = 2;
  const size_t blocks = ref_blocks(width, rows, 1);
  int sawPlain = 0, sawList = 0, sawBitmap = 0;
  for (int pattern = SMOOTH; pattern <= MANY_OUTLIERS; ++pattern) {
    fill(n, width, elt, pattern);
    size_t size = 0;
    CHECK(planar_zigzag_pfor_encode(got, sizeof(got), &size, source, width, n,
                                    elt, 1) == 0);
    size_t at = 3 * blocks;
    for (size_t k = 0; k < blocks; ++k) {
      const uint8_t *h = got + 3 * k;
      const unsigned b = h[0], nexc = h[1], format = h[2];
      const size_t positions = nexc ? ((format >> 7) ? nexc : 32) : 0;
      const size_t restBytes =
          (nexc ? positions + (nexc * (format & 0x7Fu) + 7) / 8 : 0) +
          32 * (size_t)b;
      memcpy(want, got, size);
      if (nexc == 0 && !sawPlain) {
        want[3 * k + 2] = 1;
        CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, 1, want,
                                        size) != 0);
        memcpy(want, got, size);
        want[3 * k] = 17;
        CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, 1, want,
                                        size) != 0);
        sawPlain = 1;
      } else if (nexc >= 2 && (format >> 7) && !sawList) {
        const uint8_t x = want[at];
        want[at] = want[at + 1];
        want[at + 1] = x;
        CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, 1, want,
                                        size) != 0);
        sawList = 1;
      } else if (nexc != 0 && !(format >> 7) && !sawBitmap) {
        int i = 0;
        while ((want[at + i / 8] >> (i % 8)) & 1)
          ++i;
        want[at + i / 8] |= (uint8_t)(1u << (i % 8));
        CHECK(planar_zigzag_pfor_decode(decoded, width, n, elt, 1, want,
                                        size) != 0);
        sawBitmap = 1;
      }
      at += restBytes;
    }
    CHECK(at == size);
  }
  CHECK(sawPlain && sawList && sawBitmap);
}

int main(void) {
  static const size_t geoms[][3] = {
      {1, 1, 1},
      {5, 1, 1},
      {300, 1, 1},
      {1, 40, 1},
      {7, 3, 1},
      {16, 16, 1},
      {17, 20, 1},
      {33, 15, 2},
      {100, 37, 1},
      {64, 48, 3},
      {257, 2, 1},
      {2, 300, 1},
      {176, 176, 1},
      {31, 31, 1},
      {8, 8, 4},
      {128, 23, 1},
      {255, 9, 2},
      {1024, 3, 1},
      {1025, 2, 1},
      {1030, 20, 1},
      {2048, 16, 1},
      {3000, 1, 2},
      {5000, 1, 1},
      {2100, 17, 1},
  };
  const size_t elts[] = {1, 2, 4, 8};
  for (size_t g = 0; g < sizeof(geoms) / sizeof(geoms[0]); ++g)
    for (size_t e = 0; e < 4; ++e)
      for (int pattern = 0; pattern < PATTERNS; ++pattern)
        trip(geoms[g][0], geoms[g][1], (uint32_t)geoms[g][2], elts[e], pattern);
  for (size_t e = 0; e < 4; ++e) {
    truncations(19, 21, elts[e]);
    truncations(300, 1, elts[e]);
  }
  forgeries();

  size_t written = 0;
  CHECK(planar_zigzag_pfor_bound(256, 2, 0, 1) == 0);
  CHECK(planar_zigzag_pfor_bound(256, 3, 16, 1) == 0);
  CHECK(planar_zigzag_pfor_bound(255, 2, 16, 1) == 0);
  CHECK(planar_zigzag_pfor_bound(256, 2, 16, 3) == 0);
  CHECK(planar_zigzag_pfor_encode(got, sizeof(got), &written, source, 0, 256, 2,
                                  1) != 0);
  CHECK(planar_zigzag_pfor_encode(got, sizeof(got), &written, source, 16, 255,
                                  2, 1) != 0);
  CHECK(planar_zigzag_pfor_encode(got, sizeof(got), &written, source, 16, 256,
                                  3, 1) != 0);
  CHECK(planar_zigzag_pfor_encode(got, 10, &written, source, 16, 256, 2, 1) !=
        0);
  CHECK(planar_zigzag_pfor_decode(decoded, 0, 256, 2, 1, got, 2) != 0);
  CHECK(planar_zigzag_pfor_decode(decoded, 16, 256, 3, 1, got, 2) != 0);
  CHECK(planar_zigzag_pfor_decode(decoded, 16, 256, 2, 0, got, 2) != 0);

  if (failures) {
    printf("test_planar_zigzag_pfor: %d failures\n", failures);
    return 1;
  }
  printf("test_planar_zigzag_pfor: ok\n");
  return 0;
}
