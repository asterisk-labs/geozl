#include "pfor/decode_pfor_kernel.h"
#include "pfor/encode_pfor_kernel.h"
#include "planar_zigzag/encode_planar_zigzag_kernel.h"
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

#define MAXN 4096

static uint64_t source[MAXN], transformed[MAXN];
static uint8_t stable[MAXN * 8 + 4096], fused[MAXN * 8 + 4096];

static uint64_t rng = UINT64_C(0x243f6a8885a308d3);
static uint64_t rnd(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return rng;
}

static void put(size_t i, size_t width, uint64_t value) {
  switch (width) {
  case 1:
    ((uint8_t *)source)[i] = (uint8_t)value;
    break;
  case 2:
    ((uint16_t *)source)[i] = (uint16_t)value;
    break;
  case 4:
    ((uint32_t *)source)[i] = (uint32_t)value;
    break;
  default:
    source[i] = value;
    break;
  }
}

static void trip(size_t n, size_t rowWidth, uint32_t planes,
                 size_t eltWidth) {
  CHECK(n <= MAXN);
  for (size_t i = 0; i < n; ++i)
    put(i, eltWidth, rnd() ^ (i * UINT64_C(0x9e3779b97f4a7c15)));

  const size_t planeElts = n / planes;
  for (uint32_t plane = 0; plane < planes; ++plane) {
    const size_t at = (size_t)plane * planeElts * eltWidth;
    CHECK(planar_zigzag_encode((uint8_t *)transformed + at,
                               (const uint8_t *)source + at, rowWidth,
                               planeElts, eltWidth) == 0);
  }

  const size_t bound = pfor_bound(n, eltWidth);
  size_t stableSize = 0, fusedSize = 0;
  CHECK(pfor_encode(stable, bound, &stableSize, transformed, n, eltWidth) == 0);
  CHECK(planar_zigzag_pfor_encode(fused, bound, &fusedSize, source, rowWidth,
                                  n, eltWidth, planes) == 0);
  CHECK(fusedSize == stableSize);
  CHECK(memcmp(fused, stable, stableSize) == 0);

  void *decoded = malloc(n * eltWidth);
  CHECK(decoded != NULL);
  if (decoded != NULL) {
    memset(decoded, 0xAB, n * eltWidth);
    CHECK(planar_zigzag_pfor_decode(decoded, rowWidth, n, eltWidth, planes,
                                    fused, fusedSize) == 0);
    CHECK(memcmp(decoded, source, n * eltWidth) == 0);
    free(decoded);
  }

  for (size_t len = 0; len < fusedSize; ++len)
    CHECK(planar_zigzag_pfor_decode(transformed, rowWidth, n, eltWidth, planes,
                                    fused, len) != 0);
  fused[fusedSize] = 0;
  CHECK(planar_zigzag_pfor_decode(transformed, rowWidth, n, eltWidth, planes,
                                  fused, fusedSize + 1) != 0);
}

int main(void) {
  const size_t widths[] = {1, 2, 4, 8};
  for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
    const size_t elt = widths[i];
    trip(13 * 17, 17, 1, elt);
    trip(2 * 13 * 17, 17, 2, elt);
    trip(3 * 17 * 19, 19, 3, elt);
    trip(2 * 257, 257, 1, elt);
    trip(1024, 1, 1, elt);
  }

  size_t written = 0;
  CHECK(planar_zigzag_pfor_encode(fused, sizeof(fused), &written, source, 0,
                                  256, 2, 1) != 0);
  CHECK(planar_zigzag_pfor_encode(fused, sizeof(fused), &written, source, 16,
                                  255, 2, 1) != 0);
  CHECK(planar_zigzag_pfor_encode(fused, sizeof(fused), &written, source, 16,
                                  256, 3, 1) != 0);
  CHECK(planar_zigzag_pfor_decode(transformed, 0, 256, 2, 1, fused, 2) != 0);
  CHECK(planar_zigzag_pfor_decode(transformed, 16, 256, 3, 1, fused, 2) != 0);
  CHECK(planar_zigzag_pfor_decode(transformed, 16, 256, 2, 0, fused, 2) != 0);

  if (failures) {
    printf("test_planar_zigzag_pfor: %d failures\n", failures);
    return 1;
  }
  printf("test_planar_zigzag_pfor: ok\n");
  return 0;
}
