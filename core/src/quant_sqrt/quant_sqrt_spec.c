#include "quant_sqrt_spec.h"

#include "quant_sqrt_dtype.h"
#include "quant_sqrt_half.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Terms a float reconstruction picks up when the decoder rebuilds in float
// rather than double. The step rounds, the index converts, the product rounds,
// the square doubles what came before, and the subtraction rounds again. Eight
// is the loose count of those, and measurement puts the real worst near a third
// of it, which is the direction to be wrong in.
#define QSQ_F32_TERMS 8.0

// Share of the budget the float path may give up. Past it the step it costs is no
// longer worth the speed it buys, and the double path is taken.
#define QSQ_F32_SHARE 0.25

static int fail(char *err, size_t errSize, const char *fmt, ...) {
  if (err != NULL && errSize != 0) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errSize, fmt, ap);
    va_end(ap);
  }
  return 1;
}

// Has to be consumed whole, so "1.0.0" fails instead of reading as 1.0.
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

int quant_sqrt_parse(const char *s, quant_sqrt_spec *out, char *err,
                     size_t errSize) {
  memset(out, 0, sizeof(*out));
  out->store = QUANT_SQRT_STORE_INDEX;

  if (s == NULL || strncmp(s, "SQRT:", 5) != 0)
    return fail(err, errSize,
                "quant_sqrt takes \"SQRT:MAX_ERROR=VN\", got \"%s\"",
                s == NULL ? "" : s);

  const char *cur = s + 5;
  const char *vb, *ve;
  int haveError = 0, haveA = 0, haveB = 0;

  for (;;) {
    if (keyed(&cur, "MAX_ERROR", &vb, &ve) == 0) {
      // The N is required. Without it a MAX_ERROR of 0.5 reads as half a count,
      // which is a plausible thing to mean, and the frame would come back holding
      // a bound a hundred times tighter than the caller asked for.
      double v;
      if (haveError || ve == vb || ve[-1] != 'N' || number(vb, ve - 1, &v))
        return fail(err, errSize,
                    "error \"%s\": MAX_ERROR takes one number ending in N, "
                    "which counts sigmas of noise",
                    s);
      if (!(v > 0.0))
        return fail(err, errSize, "error \"%s\": MAX_ERROR must be positive", s);
      out->k = v;
      haveError = 1;
    } else if (keyed(&cur, "A", &vb, &ve) == 0) {
      if (haveA || number(vb, ve, &out->a))
        return fail(err, errSize, "error \"%s\": A takes one number", s);
      if (!(out->a >= 0.0))
        return fail(err, errSize,
                    "error \"%s\": A is a variance at zero signal and cannot be "
                    "negative",
                    s);
      haveA = 1;
    } else if (keyed(&cur, "B", &vb, &ve) == 0) {
      if (haveB || number(vb, ve, &out->b))
        return fail(err, errSize, "error \"%s\": B takes one number", s);
      if (!(out->b > 0.0))
        return fail(err, errSize,
                    "error \"%s\": B must be positive; with B at zero the noise "
                    "does not grow with the signal and this curve is flat",
                    s);
      haveB = 1;
    } else if (keyed(&cur, "STORE", &vb, &ve) == 0) {
      const size_t n = (size_t)(ve - vb);
      if (n == 5 && strncmp(vb, "INDEX", 5) == 0)
        out->store = QUANT_SQRT_STORE_INDEX;
      else if (n == 6 && strncmp(vb, "VALUES", 6) == 0)
        out->store = QUANT_SQRT_STORE_VALUES;
      else
        return fail(err, errSize, "error \"%s\": STORE takes INDEX or VALUES", s);
    } else {
      return fail(err, errSize,
                  "error \"%s\": unknown key, expected MAX_ERROR, A, B or STORE",
                  s);
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
  if (haveA != haveB)
    return fail(err, errSize,
                "error \"%s\": A and B travel together; give both or neither "
                "and let the fit find them",
                s);
  out->have_ab = (unsigned char)haveA;
  return 0;
}

// The curve, wherever it came from. Both callers want the same two numbers and
// neither should care which branch produced them.
static int curve_of(const quant_sqrt_spec *sp, const quant_sqrt_noise *ft,
                    double *a, double *b) {
  if (sp->have_ab) {
    *a = sp->a;
    *b = sp->b;
    return 0;
  }
  if (ft == NULL || !ft->ok || !(ft->b > 0.0))
    return 1;
  *a = ft->a < 0.0 ? 0.0 : ft->a;
  *b = ft->b;
  return 0;
}

double quant_sqrt_bound(const quant_sqrt_spec *sp, const quant_sqrt_noise *ft,
                        double x) {
  double a, b;
  if (curve_of(sp, ft, &a, &b) != 0)
    return 0.0;
  const double v = a + b * fabs(x);
  return v > 0.0 ? sp->k * sqrt(v) : 0.0;
}

int quant_sqrt_resolve(const quant_sqrt_spec *sp, int dtype,
                       const quant_sqrt_stats *sc, const quant_sqrt_noise *ft,
                       quant_sqrt_params *out, char *err, size_t errSize) {
  memset(out, 0, sizeof(*out));
  if (!QSQ_DTYPE_OK(dtype))
    return fail(err, errSize, "dtype %d is not a type this codec knows", dtype);
  if (!(sp->k > 0.0))
    return fail(err, errSize, "MAX_ERROR must be positive");

  double a, b;
  if (curve_of(sp, ft, &a, &b) != 0) {
    if (ft == NULL)
      return fail(err, errSize,
                  "this recipe carries no A and B, so it needs a curve from "
                  "quant_sqrt_fit");
    return fail(err, errSize,
                "the fit did not hold: %d blocks over %d bins, range %.1fx, "
                "collinearity %.1f, residual %.3f",
                ft->blocks, ft->bins, ft->range, ft->colin, ft->resid);
  }

  // k*sqrt(a + b*x) is k*sqrt(b) * sqrt(x + a/b), so the whole curve is two
  // numbers and those two are what the header carries.
  const double c = sp->k * sqrt(b);
  const double offset = a / b;
  if (!isfinite(c) || !(c > 0.0) || !isfinite(offset) || !(offset >= 0.0))
    return fail(err, errSize,
                "a MAX_ERROR of %g with A=%g and B=%g does not give a curve",
                sp->k, a, b);
  out->offset = offset;

  // Nothing below -offset has a bound, since the model puts the variance there
  // at or under zero.
  if (sc->lo < -offset)
    return fail(err, errSize,
                "a shot bound is defined at or above -A/B, which is %g, and "
                "this raster reaches %g",
                -offset, sc->lo);
  if (!sc->anyNegative)
    out->flags |= QUANT_SQRT_FLAG_NONNEGATIVE;

  const int isInt = dtype <= QSQ_LAST_INT;
  const int values = isInt || sp->store == QUANT_SQRT_STORE_VALUES;
  const double maxAbs = fmax(fabs(sc->lo), fabs(sc->hi));

  if (values) {
    // The reconstruction has to land on a whole number, which costs half a unit
    // on top of the grid. Give the grid half the budget and the rounding the
    // other half and the two meet exactly: the grid stops biting below
    // sqrt(x+offset) = 1/(2*step), and half a unit fits under the remaining
    // c - step there when step is c/2.
    out->step = 0.5 * c;
    out->flags |= QUANT_SQRT_FLAG_STORE_VALUES;

    if (!isInt) {
      // An integer sample is already whole and rides the crossover for free. A
      // float one is not, so the whole raster has to sit above it.
      const double need = 1.0 / c;          // sqrt(x + offset) has to reach this
      const double xmin = need * need - offset;
      if (sc->hi >= sc->lo && sc->lo < xmin)
        return fail(err, errSize,
                    "STORE=VALUES rounds to whole numbers, which holds this "
                    "bound only at or above %g, and this raster reaches %g",
                    xmin, sc->lo);

      const double lim = quant_sqrt_exact_int(dtype);
      if (maxAbs > 0.0) {
        const double top = ceil(maxAbs);
        if (top > lim)
          return fail(err, errSize,
                      "STORE=VALUES needs a reconstruction exact in the output "
                      "type, and %g is past the %g it carries",
                      top, lim);
      }
    }
    return 0;
  }

  // The index path. Storing the reconstruction at the output width rounds it by a
  // fraction of its own magnitude, and that rounding bites at the top of the
  // raster, because it grows like x while the bound only grows like sqrt(x).
  const double eps = quant_sqrt_eps(dtype);
  const double u = sqrt(maxAbs + offset);
  const double chargeWide = u > 0.0 ? eps * maxAbs / u : 0.0;

  double charge = chargeWide;
  if (dtype == QSQ_F32) {
    const double chargeNarrow = QSQ_F32_TERMS * eps * u;
    if (chargeNarrow <= QSQ_F32_SHARE * c) {
      charge = chargeNarrow;
      out->flags |= QUANT_SQRT_FLAG_DECODE_F32;
    }
  }

  out->step = c - charge;
  if (!(out->step > 0.0))
    return fail(err, errSize,
                "this bound is at or below the rounding of the output type at "
                "%g, where the type resolves %g and the bound asks for %g",
                maxAbs, chargeWide, c);

  const double q = u / out->step;
  if (!(q < quant_sqrt_stream_max(dtype)))
    return fail(err, errSize,
                "this bound needs more levels than a %zu-byte index carries",
                quant_sqrt_width(dtype));
  return 0;
}

// Identical satisfies any bound, which is what carries every sample the grid
// rebuilds exactly.
#define QSQ_VER(RA, RB)                                                        \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RA), y = (double)(RB);                         \
      if (!isfinite(x) || x == y)                                              \
        continue;                                                              \
      const double v = a + b * fabs(x);                                        \
      if (!(v > 0.0))                                                          \
        continue;                                                              \
      const double r = fabs(x - y) / (sp->k * sqrt(v));                        \
      if (r > w)                                                               \
        w = r;                                                                 \
    }                                                                          \
  } while (0)

int quant_sqrt_verify(const void *src, const void *dec,
                      const quant_sqrt_spec *sp, const quant_sqrt_noise *ft,
                      int dtype, size_t nbElts, double *worst) {
  double w = 0.0, a, b;
  if (!QSQ_DTYPE_OK(dtype) || !(sp->k > 0.0) || curve_of(sp, ft, &a, &b) != 0)
    return 1;

  switch ((qsq_dtype)dtype) {
  case QSQ_U8:
    QSQ_VER(((const uint8_t *)src)[i], ((const uint8_t *)dec)[i]);
    break;
  case QSQ_U16:
    QSQ_VER(((const uint16_t *)src)[i], ((const uint16_t *)dec)[i]);
    break;
  case QSQ_U32:
    QSQ_VER(((const uint32_t *)src)[i], ((const uint32_t *)dec)[i]);
    break;
  case QSQ_U64:
    QSQ_VER(((const uint64_t *)src)[i], ((const uint64_t *)dec)[i]);
    break;
  case QSQ_I8:
    QSQ_VER(((const int8_t *)src)[i], ((const int8_t *)dec)[i]);
    break;
  case QSQ_I16:
    QSQ_VER(((const int16_t *)src)[i], ((const int16_t *)dec)[i]);
    break;
  case QSQ_I32:
    QSQ_VER(((const int32_t *)src)[i], ((const int32_t *)dec)[i]);
    break;
  case QSQ_I64:
    QSQ_VER(((const int64_t *)src)[i], ((const int64_t *)dec)[i]);
    break;
  case QSQ_F16:
    QSQ_VER(quant_sqrt_half_to_float(((const uint16_t *)src)[i]),
            quant_sqrt_half_to_float(((const uint16_t *)dec)[i]));
    break;
  case QSQ_F32:
    QSQ_VER(((const float *)src)[i], ((const float *)dec)[i]);
    break;
  default:
    QSQ_VER(((const double *)src)[i], ((const double *)dec)[i]);
    break;
  }

  if (worst != NULL)
    *worst = w;
  return 0;
}
