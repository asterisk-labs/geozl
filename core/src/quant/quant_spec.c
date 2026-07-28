#include "quant_spec.h"

#include <errno.h>
#include <stdarg.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *err, size_t n, const char *fmt, ...) {
  if (err != NULL && n != 0) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
  }
  return 1;
}

// strtod that insists on consuming the whole field, so "abs:1.0.0" and
// "abs:nan" are rejected rather than silently truncated.
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

// One "name=value" pair out of the shot argument list.
static int keyed(const char **cur, const char *name, double *out) {
  const char *s = *cur;
  const size_t n = strlen(name);
  if (strncmp(s, name, n) != 0 || s[n] != '=')
    return 1;
  const char *end = field_end(s + n + 1);
  if (number(s + n + 1, end, out))
    return 1;
  *cur = (*end == ',') ? end + 1 : end;
  return 0;
}

int quant_spec_parse(const char *s, quant_spec *out, char *err, size_t errSize) {
  memset(out, 0, sizeof(*out));
  if (s == NULL || s[0] == '\0') {
    out->mode = QUANT_SPEC_LOSSLESS;
    return 0;
  }

  if (strncmp(s, "abs:", 4) == 0) {
    const char *end = field_end(s + 4);
    if (number(s + 4, end, &out->abs_err) || *end != '\0')
      return fail(err, errSize, "error \"%s\": abs takes one number", s);
    if (out->abs_err <= 0.0)
      return fail(err, errSize, "error \"%s\": the bound must be positive", s);
    out->mode = QUANT_SPEC_EXPLICIT;
    out->curve = QUANT_CURVE_LINEAR;
    return 0;
  }

  if (strncmp(s, "rel:", 4) == 0) {
    const char *end = field_end(s + 4);
    if (end == s + 4 || end[-1] != '%')
      return fail(err, errSize,
                  "error \"%s\": rel takes a percentage, e.g. \"rel:1%%\"", s);
    double pct;
    if (number(s + 4, end - 1, &pct) || *end != '\0')
      return fail(err, errSize, "error \"%s\": rel takes one number", s);
    if (pct <= 0.0 || pct >= 100.0)
      return fail(err, errSize,
                  "error \"%s\": a relative bound is above 0%% and below 100%%",
                  s);
    out->rel_err = pct / 100.0;
    out->mode = QUANT_SPEC_EXPLICIT;
    out->curve = QUANT_CURVE_LOG;
    return 0;
  }

  if (strncmp(s, "shot:", 5) == 0) {
    const char *cur = s + 5;
    if (keyed(&cur, "a", &out->shot_a) || keyed(&cur, "b", &out->shot_b) ||
        keyed(&cur, "k", &out->shot_k) || *cur != '\0')
      return fail(err, errSize,
                  "error \"%s\": shot takes a=A,b=B,k=K in that order", s);
    if (out->shot_a < 0.0 || out->shot_b <= 0.0 || out->shot_k <= 0.0)
      return fail(err, errSize,
                  "error \"%s\": shot needs a >= 0, b > 0 and k > 0", s);
    out->mode = QUANT_SPEC_EXPLICIT;
    out->curve = QUANT_CURVE_SQRT;
    return 0;
  }

  return fail(err, errSize,
              "unknown error \"%s\"; expected abs:V, rel:P%% or shot:a=A,b=B,k=K",
              s);
}

double quant_index_max(int dtype) {
  switch (dtype) {
  case Q_U8:
    return 255.0;
  case Q_U16:
    return 65535.0;
  case Q_U32:
    return 4294967295.0;
  case Q_U64:
    return 18446744073709549568.0;
  case Q_I8:
    return 127.0;
  case Q_I16:
  case Q_F16:
    return 32767.0;
  case Q_I32:
  case Q_F32:
    return 2147483647.0;
  default:
    return 9223372036854774784.0;
  }
}

int quant_spec_resolve(const quant_spec *sp, int dtype, double minAbs,
                       double maxAbs, int anyNegative, quant_params *out,
                       char *err, size_t errSize) {
  memset(out, 0, sizeof(*out));
  out->curve = QUANT_CURVE_LINEAR;
  if (sp->mode == QUANT_SPEC_LOSSLESS)
    return 0;

  switch (sp->curve) {
  case QUANT_CURVE_LINEAR:
    out->step = 2.0 * sp->abs_err;
    // Integers carry the reconstruction, so the decoder only copies. A float
    // reconstruction is not the integer stream the codec emits, so floats keep
    // the index.
    if (dtype <= Q_LAST_INT)
      out->flags |= QUANT_FLAG_STORE_VALUES;
    return 0;

  case QUANT_CURVE_SQRT:
    if (anyNegative)
      return fail(err, errSize,
                  "shot noise is defined on non-negative data, this tile has "
                  "negative samples");
    out->curve = QUANT_CURVE_SQRT;
    out->step = sp->shot_k * sqrt(sp->shot_b);
    out->offset = sp->shot_a / sp->shot_b;
    break;

  case QUANT_CURVE_LOG: {
    out->curve = QUANT_CURVE_LOG;
    if (!isfinite(maxAbs) || maxAbs <= 0.0) {
      out->step = quant_log_step(sp->rel_err);
      out->offset = 1.0; // nothing but zeros and NaN, any anchor decodes them
      return 0;
    }

    // Below the smallest normal the representable values sit a fixed distance
    // apart rather than a fixed fraction, so a geometric grid stops being able
    // to hit them and that whole range is carried exactly instead. On integers
    // the same thing happens below roughly 8/b, where a spacing of one is no
    // longer small against the bound.
    const double sub = quant_sub(dtype);
    double geo_start, rel_round;
    if (dtype > Q_LAST_INT) {
      geo_start = quant_normal_min(dtype);
      rel_round = quant_eps(dtype);
    } else {
      geo_start = ceil(8.0 / sp->rel_err);
      rel_round = 0.5 / geo_start;
    }
    if (minAbs >= geo_start) {
      out->nsub = 0;
      out->offset = minAbs;
      if (dtype <= Q_LAST_INT)
        rel_round = 0.5 / minAbs;
    } else {
      out->nsub = (uint64_t)(geo_start / sub) - 1u;
      out->offset = geo_start;
    }

    // The reconstruction is stored back at the output width, which rounds it
    // once more, so the grid is cut by that much and the bound the frame
    // declares still holds after the cast.
    const double b = (sp->rel_err - rel_round) / (1.0 + rel_round);
    if (b <= 0.0)
      return fail(err, errSize,
                  "a relative bound of %g is at or below the rounding of the "
                  "output type",
                  sp->rel_err);
    out->step = quant_log_step(b);
    break;
  }

  default:
    return fail(err, errSize, "unknown curve %u", (unsigned)sp->curve);
  }

  // The index keeps the sample width, so a grid finer than that width can
  // address has to be refused rather than saturated into a wrong value.
  if (isfinite(maxAbs) && maxAbs > 0.0) {
    const double top = fabs(quant_fwd(maxAbs, out));
    if (!(top <= quant_index_max(dtype)))
      return fail(err, errSize,
                  "this bound needs %.0f levels, more than a %d-byte index "
                  "holds; loosen it or widen the samples",
                  top, (dtype == Q_F64 || dtype == Q_U64 || dtype == Q_I64) ? 8
                       : (dtype == Q_F16 || dtype == Q_U16 || dtype == Q_I16)
                           ? 2
                           : (dtype == Q_U8 || dtype == Q_I8) ? 1 : 4);
  }
  return 0;
}
