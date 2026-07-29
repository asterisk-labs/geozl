// Feeds arbitrary bytes to the quant codec with no frame around it, since a
// mutated frame almost never parses far enough to reach it. The first byte
// picks between the recipe parser, a forged codec header, and a round trip that
// asserts the declared bound. An abort in the last one is a sample that came
// back outside the error the frame promised.

#include "quant/decode_quant_kernel.h"
#include "quant/encode_quant_kernel.h"
#include "quant/quant_spec.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ELTS 4096

static unsigned char src[MAX_ELTS * 8];
static unsigned char idx[MAX_ELTS * 8];
static unsigned char back[MAX_ELTS * 8];

static const size_t kWidth[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};

static double get(const void *b, int dt, size_t i) {
  switch (dt) {
  case Q_U8:
    return ((const uint8_t *)b)[i];
  case Q_U16:
    return ((const uint16_t *)b)[i];
  case Q_U32:
    return ((const uint32_t *)b)[i];
  case Q_U64:
    return (double)((const uint64_t *)b)[i];
  case Q_I8:
    return ((const int8_t *)b)[i];
  case Q_I16:
    return ((const int16_t *)b)[i];
  case Q_I32:
    return ((const int32_t *)b)[i];
  case Q_I64:
    return (double)((const int64_t *)b)[i];
  case Q_F32:
    return ((const float *)b)[i];
  case Q_F64:
    return ((const double *)b)[i];
  default:
    return 0.0;
  }
}

// Read from the spec, not the resolved parameters, which are under test.
static double declared(const quant_spec *sp, double x) {
  switch (sp->curve) {
  case QUANT_CURVE_LOG:
    return sp->rel_err * fabs(x);
  case QUANT_CURVE_SQRT:
    return sp->shot_k * sqrt(sp->shot_a + sp->shot_b * x);
  default:
    return sp->abs_err;
  }
}

static void mode_parse(const uint8_t *d, size_t n) {
  char buf[128];
  if (n >= sizeof(buf))
    n = sizeof(buf) - 1;
  memcpy(buf, d, n);
  buf[n] = '\0';

  char err[256];
  quant_spec sp;
  if (quant_spec_parse(buf, &sp, err, sizeof(err)) != 0)
    return;

  // A parse that succeeded has to leave a curve the resolver knows, or the
  // switch there falls through to its default on data the parser accepted.
  if (sp.mode != QUANT_SPEC_LOSSLESS && sp.curve > QUANT_CURVE_LOG)
    abort();

  quant_params p;
  for (int dt = Q_U8; dt <= Q_F64; ++dt)
    (void)quant_spec_resolve(&sp, dt, 1e-6, 1e6, 0, &p, err, sizeof(err));
}

static void mode_header(const uint8_t *d, size_t n) {
  if (n < 27 + 8)
    return;
  const int dtype = d[0];
  if (dtype < Q_U8 || dtype > Q_F64)
    return;

  quant_params p;
  memset(&p, 0, sizeof(p));
  p.curve = d[1];
  p.flags = d[2];
  memcpy(&p.step, d + 3, 8);
  memcpy(&p.offset, d + 11, 8);
  memcpy(&p.nsub, d + 19, 8);

  // The same checks the decode binding runs. A header that gets past them must
  // not then make the kernel read or write out of bounds.
  if (p.curve > QUANT_CURVE_LOG)
    return;
  if (!isfinite(p.step) || !isfinite(p.offset) || p.step < 0.0)
    return;
  if ((p.flags & ~(unsigned)QUANT_FLAG_STORE_VALUES) != 0)
    return;
  if ((p.flags & QUANT_FLAG_STORE_VALUES) != 0 &&
      (p.curve != QUANT_CURVE_LINEAR || dtype > Q_LAST_INT))
    return;
  if (p.curve == QUANT_CURVE_LOG && !(p.offset > 0.0))
    return;
  if (p.curve != QUANT_CURVE_LOG && p.nsub != 0)
    return;

  const size_t w = kWidth[dtype];
  size_t elts = (n - 27) / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(idx, d + 27, elts * w);
  (void)quant_decode(back, idx, &p, dtype, elts);
}

static void mode_roundtrip(const uint8_t *d, size_t n) {
  static const char *kRecipes[] = {
      "abs:0.5",   "abs:5",        "abs:1000",         "rel:0.5%",
      "rel:1%",    "rel:10.71%",   "rel:33.33%",       "shot:a=4,b=1,k=0.5",
      "shot:a=100,b=1,k=0.5",      "shot:a=1,b=1,k=2", "rel:0.01%",
      "abs:0.001"};
  if (n < 4)
    return;
  const char *rec = kRecipes[d[0] % (sizeof(kRecipes) / sizeof(*kRecipes))];
  const int dtype = d[1] % (Q_F64 + 1);
  if (dtype == Q_F16)
    return; // no half type here, the kernel reads it as a bit pattern
  d += 2;
  n -= 2;

  const size_t w = kWidth[dtype];
  size_t elts = n / w;
  if (elts == 0)
    return;
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  memcpy(src, d, elts * w);

  char err[256];
  quant_spec sp;
  quant_params p;
  if (quant_spec_parse(rec, &sp, err, sizeof(err)) != 0)
    abort(); // the table is fixed, a failure here is a parser regression

  double lo, hi;
  int neg;
  quant_scan(src, dtype, elts, &lo, &hi, &neg);
  if (quant_spec_resolve(&sp, dtype, lo, hi, neg, &p, err, sizeof(err)) != 0)
    return; // refused, which is a valid answer

  // The same loop the encode binding runs, or this would test a contract the
  // codec no longer claims. What is left to check is whether quant_verify was
  // right, which the loop below does without it.
  int held = 0;
  double worst = 0.0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (quant_encode(idx, src, &p, dtype, elts) ||
        quant_decode(back, idx, &p, dtype, elts))
      abort(); // resolved parameters the kernels then reject
    if (!quant_verify(src, back, &sp, dtype, elts, &worst)) {
      held = 1;
      break;
    }
    if (!isfinite(worst) || !(worst > 1.0))
      break;
    p.step /= worst * 1.02;
    if (!(p.step > 0.0))
      break;
  }
  if (!held)
    return; // refused rather than written, which is a valid answer

  const double vlo = quant_value_lo(dtype), vhi = quant_value_hi(dtype);
  for (size_t i = 0; i < elts; ++i) {
    const double x = get(src, dtype, i), y = get(back, dtype, i);
    if (!isfinite(x))
      continue;
    // A 64 bit integer does not survive a double, so anything measured about
    // one past that point measures this harness rather than the codec.
    if ((dtype == Q_U64 || dtype == Q_I64) &&
        (fabs(x) > 9007199254740992.0 || fabs(y) > 9007199254740992.0))
      continue;
    if (y < vlo || y > vhi)
      abort(); // a reconstruction outside the type it is stored in
    if (x == y)
      continue; // exact is always inside any bound
    if (fabs(x - y) > declared(&sp, x))
      abort(); // quant_verify said this frame holds and it does not
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
    mode_header(data, size);
    break;
  default:
    mode_roundtrip(data, size);
    break;
  }
  return 0;
}