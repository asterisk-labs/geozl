#include "quant_sqrt/decode_quant_sqrt_kernel.h"
#include "quant_sqrt/encode_quant_sqrt_kernel.h"
#include "quant_sqrt/quant_sqrt_dtype.h"
#include "quant_sqrt/quant_sqrt_half.h"
#include "quant_sqrt/quant_sqrt_spec.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ELTS 4096

static unsigned char src[MAX_ELTS * 8];
static unsigned char idx[MAX_ELTS * 8];
static unsigned char back[MAX_ELTS * 8];

static double get(const void *b, int dt, size_t i) {
  switch (dt) {
  case QSQ_U8:
    return ((const uint8_t *)b)[i];
  case QSQ_U16:
    return ((const uint16_t *)b)[i];
  case QSQ_U32:
    return ((const uint32_t *)b)[i];
  case QSQ_U64:
    return (double)((const uint64_t *)b)[i];
  case QSQ_I8:
    return ((const int8_t *)b)[i];
  case QSQ_I16:
    return ((const int16_t *)b)[i];
  case QSQ_I32:
    return ((const int32_t *)b)[i];
  case QSQ_I64:
    return (double)((const int64_t *)b)[i];
  case QSQ_F16:
    return quant_sqrt_half_to_float(((const uint16_t *)b)[i]);
  case QSQ_F32:
    return ((const float *)b)[i];
  default:
    return ((const double *)b)[i];
  }
}

// A second copy of the curve on purpose, since checking quant_sqrt_bound against
// its own idea of the bound would agree with it however wrong it is. Signed x.
static double declared(const quant_sqrt_spec *sp, double x) {
  const double v = sp->a + sp->b * x;
  return v > 0.0 ? sp->k * sqrt(v) : 0.0;
}

static void mode_parse(const uint8_t *d, size_t n) {
  char buf[128];
  if (n >= sizeof(buf))
    n = sizeof(buf) - 1;
  memcpy(buf, d, n);
  buf[n] = '\0';

  char err[256];
  quant_sqrt_spec sp;
  if (quant_sqrt_parse(buf, &sp, err, sizeof(err)) != 0)
    return;

  // A parse that succeeded has to leave something the resolver can read.
  if (!(sp.k > 0.0) || !isfinite(sp.k))
    abort();
  if (sp.store != QUANT_SQRT_STORE_INDEX && sp.store != QUANT_SQRT_STORE_VALUES)
    abort();
  if (sp.have_ab && (!(sp.b > 0.0) || !(sp.a >= 0.0) || !isfinite(sp.a) ||
                     !isfinite(sp.b)))
    abort();
  // A and B travel together.
  if (!sp.have_ab && (sp.a != 0.0 || sp.b != 0.0))
    abort();

  quant_sqrt_stats sc;
  sc.lo = 0.0;
  sc.hi = 1e6;
  sc.anyNegative = 0;
  sc.anyNonFinite = 0;

  quant_sqrt_params p;
  for (int dt = QSQ_U8; dt <= QSQ_F64; ++dt) {
    if (quant_sqrt_resolve(&sp, dt, &sc, NULL, &p, err, sizeof(err)) != 0)
      continue;
    // A resolver that cuts a grid its own kernels refuse is the failure here.
    if (quant_sqrt_encode(idx, src, &p, dt, 8) != 0)
      abort();
    if (quant_sqrt_decode(back, idx, &p, dt, 8) != 0)
      abort();
  }
}

static void mode_block(const uint8_t *d, size_t n) {
  if (n < 18)
    return;
  const int dtype = d[0] % (QSQ_F64 + 2); // one past the table, so the range check runs

  quant_sqrt_params p;
  memset(&p, 0, sizeof(p));
  p.flags = d[1];
  memcpy(&p.step, d + 2, 8);
  memcpy(&p.offset, d + 10, 8);

  d += 18;
  n -= 18;
  if (dtype > QSQ_F64)
    return;
  const size_t w = quant_sqrt_width(dtype);
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);
  memcpy(idx, d, elts * w);

  // A block one end takes and the other refuses is a frame written and unreadable.
  const int e = quant_sqrt_encode(back, src, &p, dtype, elts);
  const int r = quant_sqrt_decode(back, idx, &p, dtype, elts);
  if ((e == 0) != (r == 0))
    abort();
  if (r != 0)
    return;

  // Every sample inside the type it claims, whatever the stream held. The integer
  // path is a copy, so it says nothing here.
  if (dtype <= QSQ_LAST_INT)
    return;
  const double vhi = quant_sqrt_value_hi(dtype);
  const double vlo = (p.flags & QUANT_SQRT_FLAG_NONNEGATIVE) != 0
                         ? 0.0
                         : quant_sqrt_value_lo(dtype);
  for (size_t i = 0; i < elts; ++i) {
    const double y = get(back, dtype, i);
    if (!isfinite(y))
      abort();
    if (y < vlo || y > vhi)
      abort();
  }
}

static void mode_roundtrip(const uint8_t *d, size_t n) {
  static const char *kRecipes[] = {
      "SQRT:MAX_ERROR=0.5N,A=100,B=1",
      "SQRT:MAX_ERROR=1N,A=100,B=1",
      "SQRT:MAX_ERROR=2N,A=25,B=2",
      "SQRT:MAX_ERROR=0.25N,A=0,B=1",
      "SQRT:MAX_ERROR=1N,A=0,B=1",
      "SQRT:MAX_ERROR=0.1N,A=0,B=1",
      "SQRT:MAX_ERROR=5N,A=1000,B=0.5",
      "SQRT:MAX_ERROR=1N,A=100,B=1,STORE=VALUES",
      "SQRT:MAX_ERROR=0.5N,A=25,B=2,STORE=VALUES",
      "SQRT:MAX_ERROR=3N,A=1,B=1,STORE=VALUES",
      "SQRT:MAX_ERROR=1e-3N,A=1e6,B=1",
      "SQRT:MAX_ERROR=100N,A=1,B=1"};
  if (n < 4)
    return;
  const char *rec = kRecipes[d[0] % (sizeof(kRecipes) / sizeof(*kRecipes))];
  const int dtype = d[1] % (QSQ_F64 + 1);
  d += 2;
  n -= 2;

  const size_t w = quant_sqrt_width(dtype);
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);

  char err[256];
  quant_sqrt_spec sp;
  quant_sqrt_params p;
  if (quant_sqrt_parse(rec, &sp, err, sizeof(err)) != 0)
    abort(); // the table is fixed, a failure here is a parser regression

  quant_sqrt_stats sc;
  if (quant_sqrt_scan(src, dtype, elts, &sc) != 0)
    return; // nothing finite in the tile
  if (quant_sqrt_resolve(&sp, dtype, &sc, NULL, &p, err, sizeof(err)) != 0)
    return; // refused, which is a valid answer

  if (quant_sqrt_encode(idx, src, &p, dtype, elts) != 0)
    abort(); // resolved parameters the kernels then reject
  if (quant_sqrt_decode(back, idx, &p, dtype, elts) != 0)
    abort();

  // The bound alone does not catch a crossing of zero, it is wide enough there.
  if ((sc.anyNegative == 0) !=
      ((p.flags & QUANT_SQRT_FLAG_NONNEGATIVE) != 0))
    abort();
  if ((p.flags & QUANT_SQRT_FLAG_DECODE_F32) != 0 && dtype != QSQ_F32)
    abort();

  const double vhi = quant_sqrt_value_hi(dtype);
  const double vlo = (p.flags & QUANT_SQRT_FLAG_NONNEGATIVE) != 0
                         ? 0.0
                         : quant_sqrt_value_lo(dtype);
  for (size_t i = 0; i < elts; ++i) {
    const double x = get(src, dtype, i), y = get(back, dtype, i);
    if (!isfinite(x))
      continue; // the nodata codec owns these
    // A 64 bit integer does not survive a double, so anything measured about one
    // past that point measures this harness rather than the codec.
    if ((dtype == QSQ_U64 || dtype == QSQ_I64) &&
        (fabs(x) > 9007199254740992.0 || fabs(y) > 9007199254740992.0))
      continue;
    if (y < vlo || y > vhi)
      abort(); // a reconstruction outside the type it is stored in
    if (x == y)
      continue; // exact is always inside any bound
    const double bound = declared(&sp, x);
    if (!(bound > 0.0))
      continue; // below -A/B the model has no bound, and resolve refused already
    if (fabs(x - y) > bound)
      abort();
  }

  // verify has to agree with the loop above, or it only confirms the encoder.
  double worst = -1.0;
  if (quant_sqrt_verify(src, back, &sp, NULL, dtype, elts, &worst) != 0)
    abort();
  if (!(worst <= 1.0) && !(dtype == QSQ_U64 || dtype == QSQ_I64))
    abort();

  // Two frames that each hold the bound can still disagree by twice it. The index
  // path is excluded, its step reads the raster on purpose.
  const int pinned =
      dtype <= QSQ_LAST_INT || sp.store == QUANT_SQRT_STORE_VALUES;
  if (!pinned || elts < 4)
    return;
  const size_t half = elts / 2;
  for (size_t off = 0; off + half <= elts; off += half) {
    quant_sqrt_stats s2;
    quant_sqrt_params ph;
    if (quant_sqrt_scan((const char *)src + off * w, dtype, half, &s2) != 0)
      continue;
    if (quant_sqrt_resolve(&sp, dtype, &s2, NULL, &ph, err, sizeof(err)) != 0)
      continue; // a half the recipe cannot serve says nothing about the grid
    if (ph.step != p.step || ph.offset != p.offset)
      abort(); // the grid moved because the tile did
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2)
    return 0;
  const uint8_t mode = data[0] % 3;
  ++data;
  --size;
  switch (mode) {
  case 0:
    mode_parse(data, size);
    break;
  case 1:
    mode_block(data, size);
    break;
  default:
    mode_roundtrip(data, size);
    break;
  }
  return 0;
}