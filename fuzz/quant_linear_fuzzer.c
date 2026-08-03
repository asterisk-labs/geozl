#include "quant_linear/decode_quant_linear_kernel.h"
#include "quant_linear/encode_quant_linear_kernel.h"
#include "quant_linear/quant_linear_dtype.h"
#include "quant_linear/quant_linear_half.h"
#include "quant_linear/quant_linear_spec.h"

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
  case QL_U8:
    return ((const uint8_t *)b)[i];
  case QL_U16:
    return ((const uint16_t *)b)[i];
  case QL_U32:
    return ((const uint32_t *)b)[i];
  case QL_U64:
    return (double)((const uint64_t *)b)[i];
  case QL_I8:
    return ((const int8_t *)b)[i];
  case QL_I16:
    return ((const int16_t *)b)[i];
  case QL_I32:
    return ((const int32_t *)b)[i];
  case QL_I64:
    return (double)((const int64_t *)b)[i];
  case QL_F16:
    return quant_linear_half_to_float(((const uint16_t *)b)[i]);
  case QL_F32:
    return ((const float *)b)[i];
  default:
    return ((const double *)b)[i];
  }
}

static void mode_parse(const uint8_t *d, size_t n) {
  char buf[128];
  if (n >= sizeof(buf))
    n = sizeof(buf) - 1;
  memcpy(buf, d, n);
  buf[n] = '\0';

  char err[256];
  quant_linear_spec sp;
  if (quant_linear_parse(buf, &sp, err, sizeof(err)) != 0)
    return;

  // A parse that succeeded has to leave something the resolver can read, or the
  // resolver runs on data the parser let through.
  if (!(sp.max_error > 0.0) || !isfinite(sp.max_error))
    abort();
  if (sp.store != QUANT_LINEAR_STORE_INDEX &&
      sp.store != QUANT_LINEAR_STORE_VALUES)
    abort();

  quant_linear_params p;
  for (int dt = QL_U8; dt <= QL_F64; ++dt) {
    if (quant_linear_resolve(&sp, dt, 1e6, 0, &p, err, sizeof(err)) != 0)
      continue;
    // Whatever it produced, both kernels have to take it. A resolver that can
    // cut a grid its own kernels refuse is the failure this catches.
    if (quant_linear_encode(idx, src, &p, dt, 8) != 0)
      abort();
    if (quant_linear_decode(back, idx, &p, dt, 8) != 0)
      abort();
  }
}

static void mode_block(const uint8_t *d, size_t n) {
  if (n < 10)
    return;
  const int dtype = d[0] % (QL_F64 + 2); // one past the table, so the range check runs

  quant_linear_params p;
  memset(&p, 0, sizeof(p));
  p.flags = d[1];
  memcpy(&p.step, d + 2, 8);

  d += 10;
  n -= 10;
  if (dtype > QL_F64)
    return;
  const size_t w = quant_linear_width(dtype);
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);
  memcpy(idx, d, elts * w);

  // Neither end is told anything the other is not, so a block one takes and the
  // other refuses means the frame does not round trip.
  const int e = quant_linear_encode(back, src, &p, dtype, elts);
  const int r = quant_linear_decode(back, idx, &p, dtype, elts);
  if ((e == 0) != (r == 0))
    abort();
  if (r != 0)
    return;

  // A block that decoded has to leave every sample inside the type it claims,
  // whatever the stream held.
  const double vhi = quant_linear_value_hi(dtype);
  const double vlo = (p.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0
                         ? 0.0
                         : quant_linear_value_lo(dtype);
  // A 64 bit integer does not survive a double, so anything measured about one
  // past that point measures this harness rather than the codec.
  if (dtype == QL_U64 || dtype == QL_I64)
    return;
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
      "LINEAR:MAX_ERROR=0.5",       "LINEAR:MAX_ERROR=5",
      "LINEAR:MAX_ERROR=1000",      "LINEAR:MAX_ERROR=0.001",
      "LINEAR:MAX_ERROR=2,STORE=VALUES",
      "LINEAR:MAX_ERROR=0.5,STORE=VALUES",
      "LINEAR:MAX_ERROR=50,STORE=VALUES",
      "LINEAR:MAX_ERROR=1e-4",      "LINEAR:MAX_ERROR=12.5",
      "LINEAR:MAX_ERROR=0.05"};
  if (n < 4)
    return;
  const char *rec = kRecipes[d[0] % (sizeof(kRecipes) / sizeof(*kRecipes))];
  const int dtype = d[1] % (QL_F64 + 1);
  d += 2;
  n -= 2;

  const size_t w = quant_linear_width(dtype);
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);

  char err[256];
  quant_linear_spec sp;
  quant_linear_params p;
  if (quant_linear_parse(rec, &sp, err, sizeof(err)) != 0)
    abort(); // the table is fixed, a failure here is a parser regression

  double hi;
  int neg;
  quant_linear_scan(src, dtype, elts, &hi, &neg);
  if (quant_linear_resolve(&sp, dtype, hi, neg, &p, err, sizeof(err)) != 0)
    return; // refused, which is a valid answer

  if (quant_linear_encode(idx, src, &p, dtype, elts) != 0)
    abort(); // resolved parameters the kernels then reject
  if (quant_linear_decode(back, idx, &p, dtype, elts) != 0)
    abort();

  // A tile with nothing negative in it has to come back with nothing negative,
  // and the flag that says so has to match what the scan found. The bound alone
  // does not catch a crossing of zero, since the bound at zero is wide enough to
  // cover it.
  if ((neg == 0) != ((p.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0))
    abort();

  // Read the bound off the spec, not off the resolved parameters, which are what
  // is under test. Checking quant_linear_verify against its own idea of the
  // bound would agree with it however wrong it is.
  const double vhi = quant_linear_value_hi(dtype);
  const double vlo = (p.flags & QUANT_LINEAR_FLAG_NONNEGATIVE) != 0
                         ? 0.0
                         : quant_linear_value_lo(dtype);
  for (size_t i = 0; i < elts; ++i) {
    const double x = get(src, dtype, i), y = get(back, dtype, i);
    if (!isfinite(x))
      continue; // the nodata codec owns these
    // A 64 bit integer does not survive a double, so anything measured about one
    // past that point measures this harness rather than the codec.
    if ((dtype == QL_U64 || dtype == QL_I64) &&
        (fabs(x) > 9007199254740992.0 || fabs(y) > 9007199254740992.0))
      continue;
    if (y < vlo || y > vhi)
      abort(); // a reconstruction outside the type it is stored in
    if (x == y)
      continue; // exact is always inside any bound
    if (fabs(x - y) > sp.max_error)
      abort();
  }

  // And what the codec says about itself has to match what the loop above just
  // found, or verify is only ever confirming the encoder.
  double worst = -1.0;
  if (quant_linear_verify(src, back, &sp, dtype, elts, &worst) != 0)
    abort();
  if (!(worst <= 1.0) && !(dtype == QL_U64 || dtype == QL_I64))
    abort();

  // The grid must not depend on which part of the raster this call got. Two
  // frames that each hold the bound can still disagree by twice it, and no
  // per-sample check sees that. The float index path is excluded, its step reads
  // maxAbs on purpose.
  const int pinned =
      dtype <= QL_LAST_INT || sp.store == QUANT_LINEAR_STORE_VALUES;
  if (!pinned || elts < 4)
    return;
  const size_t half = elts / 2;
  for (size_t off = 0; off + half <= elts; off += half) {
    double h2;
    int n2;
    quant_linear_params ph;
    quant_linear_scan((const char *)src + off * w, dtype, half, &h2, &n2);
    if (quant_linear_resolve(&sp, dtype, h2, n2, &ph, err, sizeof(err)) != 0)
      continue; // a half the recipe cannot serve says nothing about the grid
    if (ph.step != p.step)
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