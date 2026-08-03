#include "geozl/dtype.h"
#include "lossy/lossy_recipe.h"

#include "quant_linear/quant_linear_spec.h"
#include "quant_log/quant_log_spec.h"
#include "quant_sqrt/quant_sqrt_spec.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ELTS 4096

static unsigned char src[MAX_ELTS * 8];

// The dispatcher and the parser it delegates to have to agree in both
// directions, or a recipe behaves differently depending on which door it took.
static void mode_route(const uint8_t *d, size_t n) {
  char buf[128];
  if (n >= sizeof(buf))
    n = sizeof(buf) - 1;
  memcpy(buf, d, n);
  buf[n] = '\0';

  char err[256];
  geozl_lossy_recipe r;
  const int rc = geozl_lossy_parse(buf, &r, err, sizeof(err));

  // A refused recipe reads as lossless, so ignoring the return code is safe.
  if (rc != 0 && r.family != GEOZL_LOSSY_NONE)
    abort();

  quant_linear_spec lin;
  quant_log_spec lg;
  quant_sqrt_spec sq;
  const int okLin = quant_linear_parse(buf, &lin, err, sizeof(err)) == 0;
  const int okLog = quant_log_parse(buf, &lg, err, sizeof(err)) == 0;
  const int okSqrt = quant_sqrt_parse(buf, &sq, err, sizeof(err)) == 0;

  switch (r.family) {
  case GEOZL_LOSSY_NONE:
    if (buf[0] != '\0' && (okLin || okLog || okSqrt))
      abort();
    break;
  case GEOZL_LOSSY_LINEAR:
    if (!okLin || r.as.linear.max_error != lin.max_error ||
        r.as.linear.store != lin.store)
      abort();
    break;
  case GEOZL_LOSSY_LOG:
    if (!okLog || r.as.log.rel_err != lg.rel_err || r.as.log.store != lg.store)
      abort();
    break;
  case GEOZL_LOSSY_SQRT:
    if (!okSqrt || r.as.sqrt.k != sq.k || r.as.sqrt.have_ab != sq.have_ab ||
        r.as.sqrt.store != sq.store)
      abort();
    break;
  default:
    abort(); // a family outside the enum
  }
}

static const char *kRecipes[] = {
    "",
    "LINEAR:MAX_ERROR=0.5",
    "LINEAR:MAX_ERROR=5,STORE=VALUES",
    "LOG:MAX_ERROR=1%",
    "LOG:MAX_ERROR=0.01%",
    "SQRT:MAX_ERROR=0.5N",
    "SQRT:MAX_ERROR=0.5N,A=100,B=1",
    "SQRT:MAX_ERROR=2N,STORE=VALUES",
};
#define NB_RECIPES (sizeof(kRecipes) / sizeof(*kRecipes))

// Only width is fuzzed, since nbElts is what the buffer really holds. Half the
// inputs get one that divides, so the refusal is not the only thing measured.
static void mode_fit(const uint8_t *d, size_t n) {
  if (n < 4)
    return;
  const char *rec = kRecipes[d[0] % NB_RECIPES];
  const int dtype = d[1] % (GEOZL_DT_F64 + 1);
  const size_t width = 1 + (size_t)d[2] % 200;
  const int rectangular = (d[3] & 1) == 0;
  d += 4;
  n -= 4;

  size_t elts = n / geozl_dtype_width(dtype);
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  if (rectangular)
    elts -= elts % width;
  if (elts == 0)
    return;
  memcpy(src, d, elts * geozl_dtype_width(dtype));

  char err[256];
  geozl_lossy_recipe r, before;
  if (geozl_lossy_parse(rec, &r, err, sizeof(err)) != 0)
    abort(); // the table is fixed, a failure here is a parser regression
  before = r;

  if (geozl_lossy_fit(&r, src, dtype, width, elts, err, sizeof(err)) != 0) {
    // A refused fit leaves the recipe to be tried against another raster.
    if (memcmp(&r, &before, sizeof(r)) != 0)
      abort();
    return;
  }

  // Only a SQRT recipe with no curve of its own has anything to fill.
  if (before.family != GEOZL_LOSSY_SQRT || before.as.sqrt.have_ab) {
    if (memcmp(&r, &before, sizeof(r)) != 0)
      abort();
    return;
  }

  // quant_sqrt_fit wants a few hundred blocks, more than a fuzzer input carries,
  // so a unit test is what reaches this in practice.
  if (!r.as.sqrt.have_ab)
    abort();
  if (!isfinite(r.as.sqrt.a) || r.as.sqrt.a < 0.0)
    abort();
  if (!isfinite(r.as.sqrt.b) || !(r.as.sqrt.b > 0.0))
    abort();

  // Twice over one raster gives one grid, not two.
  geozl_lossy_recipe again = r;
  if (geozl_lossy_fit(&again, src, dtype, width, elts, err, sizeof(err)) != 0)
    abort();
  if (memcmp(&again, &r, sizeof(r)) != 0)
    abort();
}

// The chain compress_impl runs, same order, same arguments.
static void mode_chain(const uint8_t *d, size_t n) {
  if (n < 4)
    return;
  const char *rec = kRecipes[d[0] % NB_RECIPES];
  const int dtype = d[1] % (GEOZL_DT_F64 + 1);
  const size_t width = 1 + (size_t)d[2] % 200;
  d += 4;
  n -= 4;

  size_t elts = n / geozl_dtype_width(dtype);
  if (elts > MAX_ELTS)
    elts = MAX_ELTS;
  elts -= elts % width;
  if (elts == 0)
    return;
  memcpy(src, d, elts * geozl_dtype_width(dtype));

  char err[256];
  geozl_lossy_recipe r;
  geozl_lossy_plan p;
  if (geozl_lossy_parse(rec, &r, err, sizeof(err)) != 0)
    abort();
  if (geozl_lossy_fit(&r, src, dtype, width, elts, err, sizeof(err)) != 0)
    return; // no geometry or no curve, both valid answers
  if (geozl_lossy_resolve(&r, src, dtype, elts, &p, err, sizeof(err)) != 0)
    return; // refused, which is also a valid answer

  // Routing in and dispatching out are two switches. This catches them
  // disagreeing.
  if (p.family != r.family)
    abort();

  // A lossless recipe leaves nothing in the plan, so the family is enough to
  // skip the node.
  if (p.family == GEOZL_LOSSY_NONE) {
    geozl_lossy_plan zero;
    memset(&zero, 0, sizeof(zero));
    if (memcmp(&p, &zero, sizeof(p)) != 0)
      abort();
    return;
  }

  const double step = p.family == GEOZL_LOSSY_LINEAR ? p.as.linear.step
                      : p.family == GEOZL_LOSSY_LOG  ? p.as.log.step
                                                     : p.as.sqrt.step;
  if (!isfinite(step) || !(step > 0.0))
    abort();
  if (p.family == GEOZL_LOSSY_SQRT &&
      (!isfinite(p.as.sqrt.offset) || p.as.sqrt.offset < 0.0))
    abort();

  // Same raster, same grid, or compress, bench and profile disagree.
  geozl_lossy_plan again;
  if (geozl_lossy_resolve(&r, src, dtype, elts, &again, err, sizeof(err)) != 0)
    abort();
  if (memcmp(&again, &p, sizeof(p)) != 0)
    abort();
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 2)
    return 0;
  const uint8_t mode = data[0] % 3;
  ++data;
  --size;
  switch (mode) {
  case 0:
    mode_route(data, size);
    break;
  case 1:
    mode_fit(data, size);
    break;
  default:
    mode_chain(data, size);
    break;
  }
  return 0;
}