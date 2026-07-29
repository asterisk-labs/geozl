#include "quant_spec.h"
#include "decode_quant_kernel.h" // quant_decode
#include "encode_quant_kernel.h" // quant_encode
#include "quant_half.h"          // quant_half_to_float

#include <errno.h>
#include <math.h>
#include <stdarg.h>
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

int quant_spec_parse(const char *s, quant_spec *out, char *err,
                     size_t errSize) {
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

  return fail(
      err, errSize,
      "unknown error \"%s\"; expected abs:V, rel:P%% or shot:a=A,b=B,k=K", s);
}

// Every index is computed in a double, on both sides. Past the range a double
// holds exactly, adding one to an index does nothing and the nearest-index
// search stops working, so that range is the cap rather than the index width.
#define QUANT_INDEX_EXACT 9007199254740992.0 // 2^53

double quant_index_max(int dtype) {
  switch (dtype) {
  case Q_U8:
    return 255.0;
  case Q_U16:
    return 65535.0;
  case Q_U32:
    return 4294967295.0;
  case Q_U64:
    return QUANT_INDEX_EXACT;
  case Q_I8:
    return 127.0;
  case Q_I16:
  case Q_F16:
    return 32767.0;
  case Q_I32:
  case Q_F32:
    return 2147483647.0;
  default:
    return QUANT_INDEX_EXACT;
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
    // Integers carry the reconstruction so the decoder only copies, and the
    // index cannot overflow because the kernel floors the step at one. A float
    // reconstruction is not an integer stream, so floats keep the index.
    if (dtype <= Q_LAST_INT) {
      out->step = 2.0 * sp->abs_err;
      out->flags |= QUANT_FLAG_STORE_VALUES;
      return 0;
    }
    // Storing a float reconstruction rounds it by a fraction of its magnitude.
    // Nothing at ordinary values, most of the budget once the data runs large
    // against the bound, so it comes out of the step.
    out->step =
        2.0 * (sp->abs_err -
               quant_eps(dtype) * (isfinite(maxAbs) ? fabs(maxAbs) : 0.0));
    if (!(out->step > 0.0))
      return fail(err, errSize,
                  "an absolute bound of %g is at or below the rounding of the "
                  "output type at %g",
                  sp->abs_err, maxAbs);
    break;

  case QUANT_CURVE_SQRT: {
    if (anyNegative)
      return fail(err, errSize,
                  "shot noise is defined on non-negative data, this tile has "
                  "negative samples");
    out->curve = QUANT_CURVE_SQRT;
    out->offset = sp->shot_a / sp->shot_b;
    out->step = sp->shot_k * sqrt(sp->shot_b);
    if (!isfinite(maxAbs) || maxAbs <= 0.0)
      return 0; // nothing but zeros and NaN, and zero is a grid level

    // Same rounding to absorb, but the tolerance grows with sqrt(x), so an
    // integer rounding bites hardest at the bottom and a float one at the top.
    // Bottom of the range the rounding has to come out of. A positive offset
    // leaves the bound positive at zero, so zero is it. Without one the bound
    // vanishes there and only an exact carry works, so it is the smallest
    // magnitude the tile holds, which quant_scan reports finite whenever
    // maxAbs is above zero.
    const double xlo = out->offset > 0.0 ? 0.0 : minAbs;
    const double eps = quant_eps(dtype);
    const double rlo = dtype <= Q_LAST_INT ? 0.5 : eps * xlo;
    const double rhi = dtype <= Q_LAST_INT ? 0.5 : eps * maxAbs;
    double shrink = rlo / sqrt(xlo + out->offset);
    const double top = rhi / sqrt(maxAbs + out->offset);
    if (top > shrink)
      shrink = top;
    out->step -= shrink;
    if (!(out->step > 0.0))
      return fail(err, errSize,
                  "a shot bound of %g at the noise floor is at or below the "
                  "rounding of the output type",
                  sp->shot_k);
    break;
  }

  case QUANT_CURVE_LOG: {
    out->curve = QUANT_CURVE_LOG;
    if (!isfinite(maxAbs) || maxAbs <= 0.0) {
      out->step = quant_log_step(sp->rel_err);
      out->offset = 1.0; // nothing but zeros and NaN, any anchor decodes them
      return 0;
    }

    // Below the smallest normal the representable values sit a fixed distance
    // apart rather than a fixed fraction, so no geometric grid reaches them and
    // that range is carried exactly. Integers hit the same wall below 8/b.
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

    // And the grid is cut by the storage rounding, as above.
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

  // Near the top of its range f16 steps further between representable values
  // than most bounds allow, so the rounding alone breaks a bound the grid would
  // have held. Refuse rather than declare an error the frame misses.
  if (dtype > Q_LAST_INT && isfinite(maxAbs) && maxAbs > 0.0) {
    double declared;
    if (sp->curve == QUANT_CURVE_SQRT)
      declared = sp->shot_k * sqrt(sp->shot_a + sp->shot_b * maxAbs);
    else if (sp->curve == QUANT_CURVE_LOG)
      declared = sp->rel_err * maxAbs;
    else
      declared = sp->abs_err;
    if (quant_eps(dtype) * maxAbs > declared)
      return fail(err, errSize,
                  "rounding a reconstruction to this output type costs more "
                  "than the bound allows at %g",
                  maxAbs);
  }

  // The index keeps the sample width, so a grid finer than that width can
  // address has to be refused rather than saturated into a wrong value.
  if (isfinite(maxAbs) && maxAbs > 0.0) {
    const double top = fabs(quant_fwd(maxAbs, out));
    if (!(top <= quant_index_max(dtype)))
      return fail(err, errSize,
                  "this bound needs %.0f levels, more than a %d-byte index "
                  "holds; loosen it or widen the samples",
                  top, (int)quant_width(dtype));
  }
  return 0;
}
// The bound the recipe promised at x. Read from the spec, not from the resolved
// parameters, since those are the thing being checked.
static double declared_at(const quant_spec *sp, double x) {
  switch (sp->curve) {
  case QUANT_CURVE_LOG:
    return sp->rel_err * fabs(x);
  case QUANT_CURVE_SQRT:
    return sp->shot_k * sqrt(sp->shot_a + sp->shot_b * fabs(x));
  default:
    return sp->abs_err;
  }
}

// The values can be wider than a double counts, the difference cannot, so the
// subtraction happens in the wide integer.
#define Q_VER_INT(T)                                                           \
  do {                                                                         \
    const T *a = (const T *)src;                                               \
    const T *b = (const T *)dec;                                               \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const __int128 d = (__int128)a[i] - (__int128)b[i];                      \
      const double e = (double)(d < 0 ? -d : d);                               \
      if (e == 0.0)                                                            \
        continue;                                                              \
      const double bd = declared_at(sp, (double)a[i]);                         \
      const double r = bd > 0.0 ? e / bd : INFINITY;                           \
      if (r > w)                                                               \
        w = r;                                                                 \
    }                                                                          \
  } while (0)

// Identical satisfies any bound, which is what carries zero and the range no
// grid can serve.
#define Q_VER_FLT(RA, RB)                                                      \
  do {                                                                         \
    for (size_t i = 0; i < nbElts; ++i) {                                      \
      const double x = (double)(RA), y = (double)(RB);                         \
      if (!isfinite(x) || x == y)                                              \
        continue;                                                              \
      const double bd = declared_at(sp, x);                                    \
      const double r = bd > 0.0 ? fabs(x - y) / bd : INFINITY;                 \
      if (r > w)                                                               \
        w = r;                                                                 \
    }                                                                          \
  } while (0)

int quant_verify(const void *src, const void *dec, const quant_spec *sp,
                 int dtype, size_t nbElts, double *worst) {
  double w = 0.0;
  if (dtype < Q_U8 || dtype > Q_F64)
    return 1;
  if (sp->mode == QUANT_SPEC_LOSSLESS) {
    w = memcmp(src, dec, nbElts * quant_width(dtype)) == 0 ? 0.0 : INFINITY;
  } else {
    switch ((quant_dtype)dtype) {
    case Q_U8:
      Q_VER_INT(uint8_t);
      break;
    case Q_U16:
      Q_VER_INT(uint16_t);
      break;
    case Q_U32:
      Q_VER_INT(uint32_t);
      break;
    case Q_U64:
      Q_VER_INT(uint64_t);
      break;
    case Q_I8:
      Q_VER_INT(int8_t);
      break;
    case Q_I16:
      Q_VER_INT(int16_t);
      break;
    case Q_I32:
      Q_VER_INT(int32_t);
      break;
    case Q_I64:
      Q_VER_INT(int64_t);
      break;
    case Q_F16:
      Q_VER_FLT(quant_half_to_float(((const uint16_t *)src)[i]),
                quant_half_to_float(((const uint16_t *)dec)[i]));
      break;
    case Q_F32:
      Q_VER_FLT(((const float *)src)[i], ((const float *)dec)[i]);
      break;
    case Q_F64:
      Q_VER_FLT(((const double *)src)[i], ((const double *)dec)[i]);
      break;
    }
  }
  if (worst)
    *worst = w;
  return w > 1.0 ? 1 : 0;
}

int quant_fit(void *idx, void *chk, const void *src, const quant_spec *sp,
              quant_params *p, int dtype, size_t nbElts) {
  double worst = 0.0;
  // The error scales with the step on every curve, so dividing by the miss
  // converges in a step or two. It converges onto the bound and not under it,
  // and a frame landing exactly there has nothing left for the rounding that
  // follows, hence the extra 2%. Where the floor is the representation and not
  // the grid it does not converge at all, and the frame is refused.
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (quant_encode(idx, src, p, dtype, nbElts) ||
        quant_decode(chk, idx, p, dtype, nbElts))
      return -1;
    if (!quant_verify(src, chk, sp, dtype, nbElts, &worst))
      return 0;
    if (!isfinite(worst) || !(worst > 1.0))
      return 1;
    p->step /= worst * 1.02;
    if (!(p->step > 0.0))
      return 1;
  }
  return 1;
}