#include "average/decode_average_kernel.h"
#include "average/encode_average_kernel.h"
#include "binoffset/decode_binoffset_kernel.h"
#include "binoffset/encode_binoffset_kernel.h"
#include "deinterleave/decode_deinterleave_kernel.h"
#include "deinterleave/encode_deinterleave_kernel.h"
#include "delta_n/decode_delta_n_kernel.h"
#include "delta_n/encode_delta_n_kernel.h"
#include "delta_w/decode_delta_w_kernel.h"
#include "delta_w/encode_delta_w_kernel.h"
#include "floatmult/decode_floatmult_kernel.h"
#include "floatmult/encode_floatmult_kernel.h"
#include "floatquant/decode_floatquant_kernel.h"
#include "floatquant/encode_floatquant_kernel.h"
#include "intmult/decode_intmult_kernel.h"
#include "intmult/encode_intmult_kernel.h"
#include "med/decode_med_kernel.h"
#include "med/encode_med_kernel.h"
#include "nodata/decode_nodata_kernel.h"
#include "nodata/encode_nodata_kernel.h"
#include "planar/decode_planar_kernel.h"
#include "planar/encode_planar_kernel.h"
#include "wp_static/decode_wp_static_kernel.h"
#include "wp_static/encode_wp_static_kernel.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ELTS 4096

static unsigned char src[MAX_ELTS * 8];
static unsigned char mid[MAX_ELTS * 8];
static unsigned char alt[MAX_ELTS * 8];
static unsigned char back[MAX_ELTS * 8];
static uint8_t mask[MAX_ELTS];

typedef struct {
  const uint8_t *p;
  size_t n, i;
} bits;

static uint64_t take(bits *b, size_t k) {
  uint64_t v = 0;
  for (size_t j = 0; j < k; ++j)
    v |= (uint64_t)(b->i < b->n ? b->p[b->i++] : 0) << (8 * j);
  return v;
}

static void same(size_t n) {
  if (memcmp(back, src, n) != 0)
    abort();
}

// One row of a raster is at least one sample and the codecs read rows, so a
// width of zero is a geometry no encoder produces. Everything else is fair.
static void predictors(bits *b, unsigned which, size_t w, size_t n) {
  const size_t width = 1 + (size_t)take(b, 4) % (2 * MAX_ELTS);
  int e = 1;
  int16_t coeffs[4];
  uint8_t shift = 0;

  switch (which) {
  case 0:
    e = planar_encode(mid, src, width, n, w);
    break;
  case 1:
    e = med_encode(mid, src, width, n, w);
    break;
  case 2:
    e = average_encode(mid, src, width, n, w);
    break;
  case 3:
    e = delta_w_encode(mid, src, width, n, w);
    break;
  case 4:
    e = delta_n_encode(mid, src, width, n, w);
    break;
  default:
    for (int i = 0; i < 4; ++i)
      coeffs[i] = (int16_t)take(b, 2);
    shift = (uint8_t)(take(b, 1) % 64); // the binding refuses 64 and up
    e = wp_static_encode(mid, src, width, n, w, coeffs, shift);
    break;
  }
  if (e != 0)
    return; // a geometry the encoder will not take

  int d = 1;
  switch (which) {
  case 0:
    d = planar_decode(back, mid, width, n, w);
    break;
  case 1:
    d = med_decode(back, mid, width, n, w);
    break;
  case 2:
    d = average_decode(back, mid, width, n, w);
    break;
  case 3:
    d = delta_w_decode(back, mid, width, n, w);
    break;
  case 4:
    d = delta_n_decode(back, mid, width, n, w);
    break;
  default:
    d = wp_static_decode(back, mid, width, n, w, coeffs, shift);
    break;
  }

  // Both ends read the same width off the same header, so one taking a geometry
  // the other refuses means the frame the encoder just wrote cannot be read.
  if (d != 0)
    abort();
  same(n * w);
}

static void splitters(bits *b, unsigned which, size_t w, size_t n) {
  switch (which) {
  case 0:
    deinterleave_split(mid, alt, src, n, w);
    deinterleave_join(back, mid, alt, n, w);
    break;
  case 1: {
    binoffset_split(mask, mid, src, n, w);
    if (binoffset_join(back, mask, mid, n, w) != 0)
      abort();
    break;
  }
  case 2: {
    // The header carries the base in eltWidth bytes and the binding refuses one
    // below two, so those are the only bases a kernel is handed.
    const uint64_t span = w == 8 ? UINT64_MAX : ((uint64_t)1 << (8 * w)) - 1;
    const uint64_t base = 2 + take(b, 8) % (span - 1);
    intmult_split(mid, alt, src, n, w, base);
    if (intmult_join(back, mid, alt, n, w, base) != 0)
      abort();
    break;
  }
  case 3: {
    // A float codec, so the two narrow element widths are not geometries it is
    // ever handed. The encoder also picks a base it can divide by and writes it
    // into the header, so a zero one is not a frame that exists.
    if (w < 4)
      return;
    double base;
    const uint64_t bitsOf = take(b, 8);
    memcpy(&base, &bitsOf, 8);
    if (!(base > 0.0) || base > 1e300)
      return;
    floatmult_split(mid, alt, src, n, w, base, 1.0 / base);
    if (floatmult_join(back, mid, alt, n, w, base) != 0)
      abort();
    break;
  }
  case 4: {
    if (w < 4)
      return;
    const unsigned k = 1 + (unsigned)take(b, 1) % (8 * (unsigned)w - 1);
    floatquant_split(mid, alt, src, n, w, k);
    if (floatquant_join(back, mid, alt, n, w, k) != 0)
      abort();
    break;
  }
  default: {
    const size_t width = 1 + (size_t)take(b, 4) % (2 * MAX_ELTS);
    const uint64_t pattern = take(b, 8);
    nodata_mark_value(mask, src, n, w, pattern);
    nodata_fill(mid, src, mask, width, n, w);
    nodata_restore(back, mid, mask, n, w, pattern);
    break;
  }
  }
  same(n * w);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 4)
    return 0;
  bits b = {data + 1, size - 1, 0};
  const unsigned codec = data[0] % 12;

  static const size_t kWidths[] = {1, 2, 4, 8};
  const size_t w = kWidths[take(&b, 1) & 3];
  const size_t n = (size_t)take(&b, 2) % (MAX_ELTS + 1);
  if (n == 0)
    return 0;

  const size_t have = b.i < b.n ? b.n - b.i : 0;
  const size_t want = n * w;
  memset(src, 0, want);
  memcpy(src, b.p + b.i, have < want ? have : want);
  b.i += have < want ? have : want;

  memset(back, 0, want);
  if (codec < 6)
    predictors(&b, codec, w, n);
  else
    splitters(&b, codec - 6, w, n);
  return 0;
}