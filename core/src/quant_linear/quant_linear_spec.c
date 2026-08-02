#include "quant_linear_spec.h"

#include "quant_linear_dtype.h"
#include "quant_linear_half.h"

#include <errno.h>
#include <locale.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
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

// Consumed whole, so "1.0.0" fails instead of reading as 1.0. strtod reads the
// separator LC_NUMERIC declares and a recipe always writes a dot, so the dot is
// translated rather than the conversion replaced, keeping it correctly rounded.
static int number(const char *s, const char *end, double *out) {
  char buf[80];
  const size_t n = (size_t)(end - s);
  if (n == 0)
    return 1;

  const char *point = localeconv()->decimal_point;
  if (point == NULL || point[0] == '\0')
    point = ".";
  const size_t plen = strlen(point);
  if (n + plen >= sizeof(buf))
    return 1;

  size_t w = 0;
  for (size_t i = 0; i < n; ++i) {
    if (s[i] == '.') {
      memcpy(buf + w, point, plen);
      w += plen;
    } else {
      buf[w++] = s[i];
    }
  }
  buf[w] = '\0';

  char *tail = NULL;
  errno = 0;
  const double v = strtod(buf, &tail);
  // ERANGE also refuses an underflow to subnormal, which is what keeps resolve
  // from cutting a grid the kernels cannot invert.
  if (tail != buf + w || errno == ERANGE || !isfinite(v))
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

int quant_linear_parse(const char *s, quant_linear_spec *out, char *err,
                       size_t errSize) {
  memset(out, 0, sizeof(*out));
  out->store = QUANT_LINEAR_STORE_INDEX;

  if (s == NULL || strncmp(s, "LINEAR:", 7) != 0)
    return fail(err, errSize,
                "quant_linear takes \"LINEAR:MAX_ERROR=V\", got \"%s\"",
                s == NULL ? "" : s);

  const char *cur = s + 7;
  const char *vb, *ve;
  int haveError = 0;

  for (;;) {
    if (keyed(&cur, "MAX_ERROR", &vb, &ve) == 0) {
      if (haveError || number(vb, ve, &out->max_error))
        return fail(err, errSize, "error \"%s\": MAX_ERROR takes one number", s);
      if (out->max_error <= 0.0)
        return fail(err, errSize, "error \"%s\": MAX_ERROR must be positive", s);
      haveError = 1;
    } else if (keyed(&cur, "STORE", &vb, &ve) == 0) {
      const size_t n = (size_t)(ve - vb);
      if (n == 5 && strncmp(vb, "INDEX", 5) == 0)
        out->store = QUANT_LINEAR_STORE_INDEX;
      else if (n == 6 && strncmp(vb, "VALUES", 6) == 0)
        out->store = QUANT_LINEAR_STORE_VALUES;
      else
        return fail(err, errSize, "error \"%s\": STORE takes INDEX or VALUES", s);
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

int quant_linear_resolve(const quant_linear_spec *sp, int dtype, double maxAbs,
                         int anyNegative, quant_linear_params *out, char *err,
                         size_t errSize) {
  memset(out, 0, sizeof(*out));
  if (!QL_DTYPE_OK(dtype))
    return fail(err, errSize, "dtype %d is not a type this codec knows", dtype);
  if (!anyNegative)
    out->flags |= QUANT_LINEAR_FLAG_NONNEGATIVE;

  const int isInt = dtype <= QL_LAST_INT;
  const int wantValues = isInt || sp->store == QUANT_LINEAR_STORE_VALUES;

  if (wantValues) {
    // Truncated, never rounded. A grid of levels every 2V holds the bound, but
    // this one steps by a whole unit, and rounding 2V up would widen it past what
    // was declared: 0.94 asks for 1.88 and a step of 2 misses by 1.
    out->step = floor(2.0 * sp->max_error);
    if (!(out->step >= 1.0)) {
      if (isInt) {
        out->step = 1.0; // a step of one is lossless, which is the floor
      } else {
        return fail(err, errSize,
                    "STORE=VALUES needs a whole step, and a MAX_ERROR of %g "
                    "gives %g",
                    sp->max_error, 2.0 * sp->max_error);
      }
    }
    out->flags |= QUANT_LINEAR_FLAG_STORE_VALUES;

    if (!isInt) {
      // Past this a cast back would round and the bound would stop holding.
      const double top = ceil(maxAbs / out->step) * out->step;
      if (top > quant_linear_exact_int(dtype))
        return fail(err, errSize,
                    "STORE=VALUES needs a reconstruction exact in the output "
                    "type, and %g is past the %g it carries",
                    top, quant_linear_exact_int(dtype));
      if (top > quant_linear_stream_max(dtype))
        return fail(err, errSize,
                    "STORE=VALUES needs a reconstruction of %g, wider than a "
                    "%zu-byte stream carries",
                    top, quant_linear_width(dtype));
    }
    return 0;
  }

  // Two roundings, not one. The division inside nearbyint(x/step) can send q to
  // the neighbour the exact quotient would not have picked, which costs eps*|x|,
  // and storing q*step at the output width rounds again by eps*|x^|. See
  // spec.md. This is also the only thing on this path that reads the tile, and
  // STORE=VALUES is the way out.
  out->step = 2.0 * (sp->max_error - 2.0 * quant_linear_eps(dtype) * maxAbs);
  if (!(out->step > 0.0))
    return fail(err, errSize,
                "a MAX_ERROR of %g is at or below the rounding of the output "
                "type at %g",
                sp->max_error, maxAbs);
  if (maxAbs / out->step > quant_linear_stream_max(dtype))
    return fail(err, errSize,
                "a MAX_ERROR of %g needs more levels than a %zu-byte stream "
                "carries",
                sp->max_error, quant_linear_width(dtype));
  return 0;
}

// Worst error as a fraction of the declared bound, so at or under one holds. A
// non-finite source sample has no index and belongs to nodata.
#define QL_VER(RA, RB)                                                         \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RA), y = (double)(RB);                         \
      if (!isfinite(x) || x == y)                                              \
        continue;                                                              \
      const double r = fabs(x - y) / sp->max_error;                            \
      if (r > w)                                                               \
        w = r;                                                                 \
    }                                                                          \
  } while (0)

int quant_linear_verify(const void *src, const void *dec,
                        const quant_linear_spec *sp, int dtype, size_t nbElts,
                        double *worst) {
  double w = 0.0;
  if (src == NULL || dec == NULL || sp == NULL || !QL_DTYPE_OK(dtype) ||
      !(sp->max_error > 0.0))
    return 1;

  switch ((ql_dtype)dtype) {
  case QL_U8:
    QL_VER(((const uint8_t *)src)[i], ((const uint8_t *)dec)[i]);
    break;
  case QL_U16:
    QL_VER(((const uint16_t *)src)[i], ((const uint16_t *)dec)[i]);
    break;
  case QL_U32:
    QL_VER(((const uint32_t *)src)[i], ((const uint32_t *)dec)[i]);
    break;
  case QL_U64:
    QL_VER(((const uint64_t *)src)[i], ((const uint64_t *)dec)[i]);
    break;
  case QL_I8:
    QL_VER(((const int8_t *)src)[i], ((const int8_t *)dec)[i]);
    break;
  case QL_I16:
    QL_VER(((const int16_t *)src)[i], ((const int16_t *)dec)[i]);
    break;
  case QL_I32:
    QL_VER(((const int32_t *)src)[i], ((const int32_t *)dec)[i]);
    break;
  case QL_I64:
    QL_VER(((const int64_t *)src)[i], ((const int64_t *)dec)[i]);
    break;
  case QL_F16:
    QL_VER(quant_linear_half_to_float(((const uint16_t *)src)[i]),
           quant_linear_half_to_float(((const uint16_t *)dec)[i]));
    break;
  case QL_F32:
    QL_VER(((const float *)src)[i], ((const float *)dec)[i]);
    break;
  default:
    QL_VER(((const double *)src)[i], ((const double *)dec)[i]);
    break;
  }

  if (worst != NULL)
    *worst = w;
  return 0;
}
