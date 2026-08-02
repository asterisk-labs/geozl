#include "quant_log_spec.h"

#include "quant_log_dtype.h"
#include "quant_log_half.h"
#include "quant_log_math.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, size_t errSize, const char *fmt, ...) {
  if (err != NULL && errSize != 0) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errSize, fmt, ap);
    va_end(ap);
  }
  return 1;
}

// Consumed whole, so "1.0.0" fails instead of reading as 1.0.
static int number(const char *s, const char *end, double *out) {
  char buf[64];
  const size_t n = (size_t)(end - s);
  if (n == 0 || n >= sizeof(buf))
    return 1;
  memcpy(buf, s, n);
  buf[n] = '\0';
  char *tail = NULL;
  errno = 0;
  const double v = strtod(buf, &tail);
  if (tail != buf + n || errno == ERANGE || !isfinite(v))
    return 1;
  *out = v;
  return 0;
}

static const char *field_end(const char *s) {
  while (*s != '\0' && *s != ',')
    ++s;
  return s;
}

static int keyed(const char **cur, const char *name, const char **vbeg,
                 const char **vend) {
  const char *t = *cur;
  const size_t n = strlen(name);
  if (strncmp(t, name, n) != 0 || t[n] != '=')
    return 1;
  *vbeg = t + n + 1;
  *vend = field_end(*vbeg);
  *cur = *vend;
  return 0;
}

int quant_log_parse(const char *s, quant_log_spec *out, char *err,
                    size_t errSize) {
  memset(out, 0, sizeof(*out));
  out->store = QUANT_LOG_STORE_INDEX;

  if (s == NULL || strncmp(s, "LOG:", 4) != 0)
    return fail(err, errSize, "quant_log takes \"LOG:MAX_ERROR=V%%\", got \"%s\"",
                s == NULL ? "" : s);

  const char *cur = s + 4;
  const char *vb, *ve;
  int haveError = 0, haveStore = 0;

  for (;;) {
    if (keyed(&cur, "MAX_ERROR", &vb, &ve) == 0) {
      // The per cent sign is required, or MAX_ERROR=1 reads as a bound of one and
      // the caller almost certainly meant a hundredth of that.
      double v;
      if (haveError || ve == vb || ve[-1] != '%' || number(vb, ve - 1, &v))
        return fail(err, errSize,
                    "error \"%s\": MAX_ERROR takes one number ending in %%", s);
      if (!(v > 0.0))
        return fail(err, errSize, "error \"%s\": MAX_ERROR must be positive", s);
      if (!(v < 100.0))
        return fail(err, errSize,
                    "error \"%s\": a MAX_ERROR of %g%% is the value itself", s,
                    v);
      out->rel_err = v / 100.0;
      haveError = 1;
    } else if (keyed(&cur, "STORE", &vb, &ve) == 0) {
      const size_t n = (size_t)(ve - vb);
      if (haveStore)
        return fail(err, errSize, "error \"%s\": STORE is given twice", s);
      if (n == 5 && strncmp(vb, "INDEX", 5) == 0)
        out->store = QUANT_LOG_STORE_INDEX;
      else if (n == 6 && strncmp(vb, "VALUES", 6) == 0)
        out->store = QUANT_LOG_STORE_VALUES;
      else
        return fail(err, errSize, "error \"%s\": STORE takes INDEX or VALUES", s);
      haveStore = 1;
    } else {
      return fail(err, errSize,
                  "error \"%s\": unknown key, expected MAX_ERROR or STORE", s);
    }
    if (*cur == '\0')
      break;
    if (*cur != ',' || cur[1] == '\0')
      return fail(err, errSize, "error \"%s\": a key has to follow every comma",
                  s);
    ++cur;
  }
  if (!haveError)
    return fail(err, errSize, "error \"%s\": MAX_ERROR is required", s);
  return 0;
}

int quant_log_resolve(const quant_log_spec *sp, int dtype,
                      const quant_log_stats *sc, quant_log_params *out,
                      char *err, size_t errSize) {
  memset(out, 0, sizeof(*out));
  if (!QLOG_DTYPE_OK(dtype))
    return fail(err, errSize, "dtype %d is not a type this codec knows", dtype);
  if (!(sp->rel_err > 0.0))
    return fail(err, errSize, "MAX_ERROR must be positive");

  const double b = sp->rel_err;
  const int isInt = dtype <= QLOG_LAST_INT;
  const int values = isInt || sp->store == QUANT_LOG_STORE_VALUES;

  if (!sc->anyNegative)
    out->flags |= QUANT_LOG_FLAG_NONNEGATIVE;
  if (values)
    out->flags |= QUANT_LOG_FLAG_STORE_VALUES;

  const double slack = quant_log_slack(dtype, values);

  if (values) {
    // Two things stand between the sample and the reconstruction and each is a
    // ratio, so they multiply. The grid, and the rounding of the level to a whole
    // number. Split the budget evenly and each gets sqrt(1+b).
    const double r = sqrt(1.0 + b);
    const double half =
        (log1p(r - 1.0) - log1p(slack * QLOG_LN2)) * QLOG_LOG2E - slack;
    if (!(half > 0.0))
      return fail(err, errSize,
                  "a MAX_ERROR of %g%% is too tight for a whole-number "
                  "reconstruction",
                  b * 100.0);
    out->step = 2.0 * half;

    // Below this the gap between levels is under one, so the level nearest a
    // whole number rounds back to it and the reconstruction is exact. An integer
    // sample is already whole and rides that for free. A float sample is not, and
    // rounding it costs more than the bound, so it is refused rather than served.
    if (!isInt) {
      const double cross = 0.5 / (r - 1.0);
      if (sc->minAbs < cross)
        return fail(err, errSize,
                    "STORE=VALUES rounds to whole numbers, which holds a "
                    "MAX_ERROR of %g%% only at or above %g, and this tile "
                    "reaches %g",
                    b * 100.0, cross, sc->minAbs);

      const double lim = quant_log_exact_int(dtype);
      if (sc->maxAbs > 0.0) {
        const double j = ceil(log2(sc->maxAbs) / out->step);
        const double top = qlog_value_level(j, out->step);
        if (top > lim)
          return fail(err, errSize,
                      "STORE=VALUES needs a reconstruction exact in the output "
                      "type, and %g is past the %g it carries",
                      top, lim);
      }
    }
    return 0;
  }

  // The index path adds the rounding when the level reaches the output width.
  const double eps = quant_log_eps(dtype);
  const double half =
      (log1p(b) - log1p(eps) - log1p(slack * QLOG_LN2)) * QLOG_LOG2E - slack;
  if (!(half > 0.0))
    return fail(err, errSize,
                "a MAX_ERROR of %g%% is at or below what this type rebuilds to, "
                "which bottoms out near %g%%",
                b * 100.0, (eps + slack * QLOG_LN2) * 100.0);
  out->step = 2.0 * half;

  // The whole type has to fit, since the grid never reads the tile and so cannot
  // be cut down to the range that happens to be present.
  if (quant_log_index_top(out->step, dtype) >= quant_log_stream_max(dtype))
    return fail(err, errSize,
                "a MAX_ERROR of %g%% needs more levels than a %zu-byte index "
                "carries",
                b * 100.0, quant_log_width(dtype));
  return 0;
}

// Identical satisfies any bound, which is what carries zero and every sample the
// grid rebuilds exactly.
#define QLOG_VER(RA, RB)                                                       \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RA), y = (double)(RB);                         \
      if (!isfinite(x) || x == y)                                              \
        continue;                                                              \
      const double a = fabs(x);                                                \
      if (a < nmin) {                                                          \
        ++skip;                                                                \
        continue;                                                              \
      }                                                                        \
      const double r = fabs(x - y) / (sp->rel_err * a);                        \
      if (r > w)                                                               \
        w = r;                                                                 \
    }                                                                          \
  } while (0)

int quant_log_verify(const void *src, const void *dec, const quant_log_spec *sp,
                     int dtype, size_t nbElts, double *worst, size_t *skipped) {
  double w = 0.0;
  size_t skip = 0;
  if (!QLOG_DTYPE_OK(dtype) || !(sp->rel_err > 0.0))
    return 1;
  const double nmin =
      dtype > QLOG_LAST_INT ? quant_log_normal_min(dtype) : 0.0;

  switch ((qlog_dtype)dtype) {
  case QLOG_U8:
    QLOG_VER(((const uint8_t *)src)[i], ((const uint8_t *)dec)[i]);
    break;
  case QLOG_U16:
    QLOG_VER(((const uint16_t *)src)[i], ((const uint16_t *)dec)[i]);
    break;
  case QLOG_U32:
    QLOG_VER(((const uint32_t *)src)[i], ((const uint32_t *)dec)[i]);
    break;
  case QLOG_U64:
    QLOG_VER(((const uint64_t *)src)[i], ((const uint64_t *)dec)[i]);
    break;
  case QLOG_I8:
    QLOG_VER(((const int8_t *)src)[i], ((const int8_t *)dec)[i]);
    break;
  case QLOG_I16:
    QLOG_VER(((const int16_t *)src)[i], ((const int16_t *)dec)[i]);
    break;
  case QLOG_I32:
    QLOG_VER(((const int32_t *)src)[i], ((const int32_t *)dec)[i]);
    break;
  case QLOG_I64:
    QLOG_VER(((const int64_t *)src)[i], ((const int64_t *)dec)[i]);
    break;
  case QLOG_F16:
    QLOG_VER(quant_log_half_to_float(((const uint16_t *)src)[i]),
             quant_log_half_to_float(((const uint16_t *)dec)[i]));
    break;
  case QLOG_F32:
    QLOG_VER(((const float *)src)[i], ((const float *)dec)[i]);
    break;
  default:
    QLOG_VER(((const double *)src)[i], ((const double *)dec)[i]);
    break;
  }

  if (worst != NULL)
    *worst = w;
  if (skipped != NULL)
    *skipped = skip;
  return 0;
}
