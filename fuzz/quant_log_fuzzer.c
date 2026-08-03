#include "quant_log/decode_quant_log_kernel.h"
#include "quant_log/encode_quant_log_kernel.h"
#include "quant_log/quant_log_dtype.h"
#include "quant_log/quant_log_half.h"
#include "quant_log/quant_log_spec.h"

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
  case QLOG_U8:
    return ((const uint8_t *)b)[i];
  case QLOG_U16:
    return ((const uint16_t *)b)[i];
  case QLOG_U32:
    return ((const uint32_t *)b)[i];
  case QLOG_U64:
    return (double)((const uint64_t *)b)[i];
  case QLOG_I8:
    return ((const int8_t *)b)[i];
  case QLOG_I16:
    return ((const int16_t *)b)[i];
  case QLOG_I32:
    return ((const int32_t *)b)[i];
  case QLOG_I64:
    return (double)((const int64_t *)b)[i];
  case QLOG_F16:
    return quant_log_half_to_float(((const uint16_t *)b)[i]);
  case QLOG_F32:
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
  quant_log_spec sp;
  if (quant_log_parse(buf, &sp, err, sizeof(err)) != 0)
    return;

  // A parse that succeeded has to leave something the resolver can read, or the
  // resolver runs on data the parser let through.
  if (!(sp.rel_err > 0.0) || !isfinite(sp.rel_err))
    abort();
  if (sp.store != QUANT_LOG_STORE_INDEX && sp.store != QUANT_LOG_STORE_VALUES)
    abort();

  const quant_log_stats sc = {1.0e4, 1.0e4, 0, 0};
  quant_log_params p;
  for (int dt = QLOG_U8; dt <= QLOG_F64; ++dt) {
    if (quant_log_resolve(&sp, dt, &sc, &p, err, sizeof(err)) != 0)
      continue;
    // Whatever it produced, both kernels have to take it. A resolver that can
    // cut a grid its own kernels refuse is the failure this catches.
    if (quant_log_encode(idx, src, &p, dt, 8) != 0)
      abort();
    if (quant_log_decode(back, idx, &p, dt, 8) != 0)
      abort();
  }
}

static void mode_block(const uint8_t *d, size_t n) {
  if (n < 10)
    return;
  const int dtype = d[0] % (QLOG_F64 + 2); // one past the table, so the range check runs

  quant_log_params p;
  memset(&p, 0, sizeof(p));
  p.flags = d[1];
  memcpy(&p.step, d + 2, 8);

  d += 10;
  n -= 10;
  if (dtype > QLOG_F64)
    return;
  const size_t w = quant_log_width(dtype);
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);
  memcpy(idx, d, elts * w);

  // Neither end is told anything the other is not, so a block one takes and the
  // other refuses means the frame does not round trip.
  const int e = quant_log_encode(back, src, &p, dtype, elts);
  const int r = quant_log_decode(back, idx, &p, dtype, elts);
  if ((e == 0) != (r == 0))
    abort();
  if (r != 0)
    return;

  // A block that decoded has to leave every sample inside the type it claims,
  // whatever the stream held. An integer stream is the output, so it does by
  // construction.
  if (dtype <= QLOG_LAST_INT)
    return;
  const double vhi = quant_log_value_hi(dtype);
  const double vlo = (p.flags & QUANT_LOG_FLAG_NONNEGATIVE) != 0
                         ? 0.0
                         : quant_log_value_lo(dtype);
  for (size_t i = 0; i < elts; ++i) {
    const double y = get(back, dtype, i);
    if (!isfinite(y))
      abort();
    if (y < vlo || y > vhi)
      abort();
  }
}

static void mode_roundtrip(const uint8_t *d, size_t n) {
  static const char *kRecipes[] = {"LOG:MAX_ERROR=1%",
                                   "LOG:MAX_ERROR=0.5%",
                                   "LOG:MAX_ERROR=5%",
                                   "LOG:MAX_ERROR=0.1%",
                                   "LOG:MAX_ERROR=0.002%",
                                   "LOG:MAX_ERROR=25%",
                                   "LOG:MAX_ERROR=1%,STORE=VALUES",
                                   "LOG:MAX_ERROR=0.5%,STORE=VALUES",
                                   "LOG:MAX_ERROR=10%,STORE=VALUES"};
  if (n < 4)
    return;
  const char *rec = kRecipes[d[0] % (sizeof(kRecipes) / sizeof(*kRecipes))];
  const int dtype = d[1] % (QLOG_F64 + 1);
  d += 2;
  n -= 2;

  const size_t w = quant_log_width(dtype);
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);

  char err[256];
  quant_log_spec sp;
  quant_log_params p;
  if (quant_log_parse(rec, &sp, err, sizeof(err)) != 0)
    abort(); // the table is fixed, a failure here is a parser regression

  quant_log_stats sc;
  quant_log_scan(src, dtype, elts, &sc);
  if (quant_log_resolve(&sp, dtype, &sc, &p, err, sizeof(err)) != 0)
    return; // refused, which is a valid answer

  if (quant_log_encode(idx, src, &p, dtype, elts) != 0)
    abort(); // resolved parameters the kernels then reject
  if (quant_log_decode(back, idx, &p, dtype, elts) != 0)
    abort();

  // A tile with nothing negative in it has to come back with nothing negative,
  // and the flag that says so has to match what the scan found.
  if ((sc.anyNegative == 0) != ((p.flags & QUANT_LOG_FLAG_NONNEGATIVE) != 0))
    abort();

  // Read the bound off the spec, not off the resolved parameters, which are what
  // is under test. Checking quant_log_verify against its own idea of the bound
  // would agree with it however wrong it is.
  const double nmin =
      dtype > QLOG_LAST_INT ? quant_log_normal_min(dtype) : 0.0;
  const double vhi = quant_log_value_hi(dtype);
  const double vlo = (p.flags & QUANT_LOG_FLAG_NONNEGATIVE) != 0
                         ? 0.0
                         : quant_log_value_lo(dtype);
  const int wide = dtype == QLOG_U64 || dtype == QLOG_I64;
  for (size_t i = 0; i < elts; ++i) {
    const double x = get(src, dtype, i), y = get(back, dtype, i);
    if (!isfinite(x))
      continue; // the nodata codec owns these
    // A 64 bit integer does not survive a double, so anything measured about one
    // past that point measures this harness rather than the codec.
    if (wide && (fabs(x) > 9007199254740992.0 || fabs(y) > 9007199254740992.0))
      continue;
    if (y < vlo || y > vhi)
      abort(); // a reconstruction outside the type it is stored in
    if (x == y)
      continue; // exact is always inside any bound
    // Below the smallest normal the gap between neighbours is flat, and no grid
    // reaches down there.
    if (fabs(x) < nmin)
      continue;
    if (fabs(x - y) > sp.rel_err * fabs(x))
      abort();
  }

  // And what the codec says about itself has to match what the loop above just
  // found, or verify is only ever confirming the encoder.
  double worst = -1.0;
  size_t skipped = 0;
  if (quant_log_verify(src, back, &sp, dtype, elts, &worst, &skipped) != 0)
    abort();
  if (!wide && skipped == 0 && !(worst <= 1.0))
    abort();

  // Nothing about the grid comes from the tile, so the same recipe on any part
  // of it cuts the same step. Two frames that each hold the bound can still
  // disagree by twice it, and no per-sample check sees that.
  if (elts < 4)
    return;
  const size_t half = elts / 2;
  for (size_t off = 0; off + half <= elts; off += half) {
    quant_log_stats s2;
    quant_log_params ph;
    quant_log_scan((const char *)src + off * w, dtype, half, &s2);
    if (quant_log_resolve(&sp, dtype, &s2, &ph, err, sizeof(err)) != 0)
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