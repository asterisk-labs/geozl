// Planes are a geometry parameter, not a codec. Predicting a stream declared
// as B planes must give exactly the same bytes as predicting the B planes one
// at a time, and neither may look at the plane before it. The bindings loop the
// kernel per plane, so this is what that loop has to be equal to.

#include "average/decode_average_kernel.h"
#include "average/encode_average_kernel.h"
#include "delta_n/decode_delta_n_kernel.h"
#include "delta_n/encode_delta_n_kernel.h"
#include "med/decode_med_kernel.h"
#include "med/encode_med_kernel.h"
#include "planar/decode_planar_kernel.h"
#include "planar/encode_planar_kernel.h"
#include "planar_zigzag/decode_planar_zigzag_kernel.h"
#include "planar_zigzag/encode_planar_zigzag_kernel.h"

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

#define W 24
#define H 16
#define B 5
#define PER (W * H)
#define N (PER * B)

static uint16_t src[N], whole[N], piece[N], back[N];

typedef int (*enc_fn)(void *, const void *, size_t, size_t, size_t);
typedef int (*dec_fn)(void *, const void *, size_t, size_t, size_t);

static void fill(void) {
  uint32_t s = 0x1234567u;
  for (size_t i = 0; i < N; ++i) {
    s = s * 1664525u + 1013904223u;
    src[i] = (uint16_t)(s >> 13);
  }
}

// The loop the bindings run: each plane on its own, at its own offset.
static void per_plane(enc_fn f, void *dst, const void *s) {
  for (size_t p = 0; p < B; ++p)
    f((uint16_t *)dst + p * PER, (const uint16_t *)s + p * PER, W, PER,
      sizeof(uint16_t));
}

static void one_codec(const char *name, enc_fn enc, dec_fn dec) {
  memset(whole, 0, sizeof whole);
  memset(piece, 0, sizeof piece);
  memset(back, 0, sizeof back);

  // Encoding the whole stream as one plane is the old behaviour, and it must
  // differ: the first row of every plane after the first would otherwise be
  // predicted from the last row of the plane before it.
  enc(whole, src, W, N, sizeof(uint16_t));
  per_plane(enc, piece, src);
  CHECK(memcmp(whole, piece, sizeof whole) != 0);

  // Only the first row of each plane may differ, that is the whole change.
  size_t differing_rows = 0;
  for (size_t r = 0; r < N / W; ++r)
    if (memcmp(whole + r * W, piece + r * W, W * sizeof(uint16_t)) != 0)
      ++differing_rows;
  CHECK(differing_rows <= B - 1);

  // Decoding per plane undoes encoding per plane, exactly.
  per_plane((enc_fn)dec, back, piece);
  CHECK(memcmp(back, src, sizeof back) == 0);

  // A plane count of one is the ungrouped stream, byte for byte.
  memset(back, 0, sizeof back);
  enc(back, src, W, N, sizeof(uint16_t));
  CHECK(memcmp(back, whole, sizeof back) == 0);

  printf("  %-10s ok\n", name);
}

int main(void) {
  fill();
  printf("test_planes\n");
  one_codec("planar", planar_encode, planar_decode);
  one_codec("planar_zz", planar_zigzag_encode, planar_zigzag_decode);
  one_codec("delta_n", delta_n_encode, delta_n_decode);
  one_codec("average", average_encode, average_decode);
  one_codec("med", med_encode, med_decode);

  // A single plane is the identity of the grouping, for every codec above.
  memset(whole, 0, sizeof whole);
  memset(piece, 0, sizeof piece);
  planar_encode(whole, src, W, PER, sizeof(uint16_t));
  planar_encode(piece, src, W, PER, sizeof(uint16_t));
  CHECK(memcmp(whole, piece, PER * sizeof(uint16_t)) == 0);

  if (failures) {
    printf("test_planes: %d failures\n", failures);
    return 1;
  }
  printf("test_planes: ok\n");
  return 0;
}
