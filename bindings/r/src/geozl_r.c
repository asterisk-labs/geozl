// R bindings over the geozl_2d_* C API. R holds samples as double, integer or
// raw; pack and unpack move them to and from the native little-endian samples
// the codecs read, checking every value on the way in.

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Utils.h>

#include "geozl/geozl.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define ERR_SIZE 256

static const size_t kWidth[] = {1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8};
static const char *const kName[] = {"uint8", "uint16", "uint32", "uint64",
                                    "int8", "int16", "int32", "int64",
                                    "float16", "float32", "float64"};

static int dtype_arg(SEXP s) {
  int d = Rf_asInteger(s);
  if (!GEOZL_DT_OK(d))
    Rf_error("unknown datatype code %d", d);
  return d;
}

static const char *string_arg(SEXP s, const char *name) {
  if (!Rf_isString(s) || XLENGTH(s) != 1 || STRING_ELT(s, 0) == NA_STRING)
    Rf_error("`%s` must be a single string", name);
  return Rf_translateCharUTF8(STRING_ELT(s, 0));
}

// NULL for lossless.
static const char *error_arg(SEXP s) {
  return Rf_isNull(s) ? NULL : string_arg(s, "error");
}

static uint32_t u32_arg(SEXP s, const char *name) {
  double v = Rf_asReal(s);
  if (!R_FINITE(v) || v < 1 || v > 4294967295.0 || v != floor(v))
    Rf_error("`%s` must be a whole number from 1 to 4294967295", name);
  return (uint32_t)v;
}

static size_t reps_arg(SEXP s) {
  double v = Rf_asReal(s);
  if (!R_FINITE(v) || v < 1 || v > INT_MAX || v != floor(v))
    Rf_error("`reps` must be a whole number from 1 to %d", INT_MAX);
  return (size_t)v;
}

static const uint8_t *raw_arg(SEXP s, const char *name) {
  if (TYPEOF(s) != RAWSXP)
    Rf_error("`%s` must be a raw vector", name);
  return RAW(s);
}

static int is_int_type(int d) { return d <= GEOZL_DT_LAST_INT; }

// Smallest and largest value of an integer type, and its upper bound as an
// exclusive limit, since 2^64 - 1 and 2^63 - 1 round up as doubles.
static void int_range(int d, double *lo, double *hiExcl) {
  switch (d) {
  case GEOZL_DT_U8: *lo = 0; *hiExcl = 256.0; break;
  case GEOZL_DT_U16: *lo = 0; *hiExcl = 65536.0; break;
  case GEOZL_DT_U32: *lo = 0; *hiExcl = 4294967296.0; break;
  case GEOZL_DT_U64: *lo = 0; *hiExcl = 18446744073709551616.0; break;
  case GEOZL_DT_I8: *lo = -128.0; *hiExcl = 128.0; break;
  case GEOZL_DT_I16: *lo = -32768.0; *hiExcl = 32768.0; break;
  case GEOZL_DT_I32: *lo = -2147483648.0; *hiExcl = 2147483648.0; break;
  default: *lo = -9223372036854775808.0; *hiExcl = 9223372036854775808.0; break;
  }
}

static int int_fits(int d, double v) {
  double lo, hi;
  int_range(d, &lo, &hi);
  return R_FINITE(v) && v == floor(v) && v >= lo && v < hi;
}

// IEEE half, round to nearest even. Sets *overflow for finite values that round
// past 65504.
static uint16_t f64_to_f16(double v, int *overflow) {
  *overflow = 0;
  uint16_t sign = signbit(v) ? 0x8000u : 0;
  double a = fabs(v);
  if (ISNAN(v))
    return 0x7E00u;
  if (!R_FINITE(v))
    return sign | 0x7C00u;
  if (a < ldexp(1.0, -14)) // subnormal: units of 2^-24
    return sign | (uint16_t)nearbyint(ldexp(a, 24));
  int e;
  frexp(a, &e);
  e -= 1; // a = s * 2^e with s in [1, 2)
  double q = nearbyint(ldexp(a, 10 - e));
  if (q == 2048.0) {
    q = 1024.0;
    e += 1;
  }
  if (e > 15) {
    *overflow = 1;
    return sign | 0x7C00u;
  }
  return sign | (uint16_t)((e + 15) << 10) | (uint16_t)(q - 1024.0);
}

static double f16_to_f64(uint16_t h) {
  double sign = (h & 0x8000u) ? -1.0 : 1.0;
  int e = (h >> 10) & 0x1F;
  int m = h & 0x3FF;
  if (e == 0)
    return sign * ldexp((double)m, -24);
  if (e == 31)
    return m ? R_NaN : sign * R_PosInf;
  return sign * ldexp((double)(m + 1024), e - 25);
}

// Store v, already checked, as sample i of type d.
static void store(uint8_t *dst, R_xlen_t i, int d, double v) {
  uint8_t *p = dst + (size_t)i * kWidth[d];
  switch (d) {
  case GEOZL_DT_U8: *p = (uint8_t)v; break;
  case GEOZL_DT_I8: *(int8_t *)p = (int8_t)v; break;
  case GEOZL_DT_U16: { uint16_t t = (uint16_t)v; memcpy(p, &t, 2); break; }
  case GEOZL_DT_I16: { int16_t t = (int16_t)v; memcpy(p, &t, 2); break; }
  case GEOZL_DT_U32: { uint32_t t = (uint32_t)v; memcpy(p, &t, 4); break; }
  case GEOZL_DT_I32: { int32_t t = (int32_t)v; memcpy(p, &t, 4); break; }
  case GEOZL_DT_U64: { uint64_t t = (uint64_t)v; memcpy(p, &t, 8); break; }
  case GEOZL_DT_I64: { int64_t t = (int64_t)v; memcpy(p, &t, 8); break; }
  case GEOZL_DT_F16: { int o; uint16_t t = f64_to_f16(v, &o); memcpy(p, &t, 2); break; }
  case GEOZL_DT_F32: { float t = (float)v; memcpy(p, &t, 4); break; }
  default: memcpy(p, &v, 8); break;
  }
}

static double load(const uint8_t *src, R_xlen_t i, int d) {
  const uint8_t *p = src + (size_t)i * kWidth[d];
  switch (d) {
  case GEOZL_DT_U8: return *p;
  case GEOZL_DT_I8: return *(const int8_t *)p;
  case GEOZL_DT_U16: { uint16_t t; memcpy(&t, p, 2); return t; }
  case GEOZL_DT_I16: { int16_t t; memcpy(&t, p, 2); return t; }
  case GEOZL_DT_U32: { uint32_t t; memcpy(&t, p, 4); return t; }
  case GEOZL_DT_I32: { int32_t t; memcpy(&t, p, 4); return t; }
  case GEOZL_DT_U64: { uint64_t t; memcpy(&t, p, 8); return (double)t; }
  case GEOZL_DT_I64: { int64_t t; memcpy(&t, p, 8); return (double)t; }
  case GEOZL_DT_F16: { uint16_t t; memcpy(&t, p, 2); return f16_to_f64(t); }
  case GEOZL_DT_F32: { float t; memcpy(&t, p, 4); return t; }
  default: { double t; memcpy(&t, p, 8); return t; }
  }
}

// A sample's storage bits, zero-extended to the form the C API uses.
static uint64_t sample_bits(const uint8_t *src, R_xlen_t i, int d) {
  uint64_t bits = 0;
  memcpy(&bits, src + (size_t)i * kWidth[d], kWidth[d]);
  return bits;
}

// The sentinel's bit pattern at type d, in its low bytes.
static uint64_t nodata_bits(int d, double v) {
  if (d == GEOZL_DT_F16)
    Rf_error("a float16 raster carries no nodata sentinel; NaN still works");
  if (is_int_type(d) ? !int_fits(d, v) : (R_FINITE(v) && fabs(v) > FLT_MAX && d == GEOZL_DT_F32))
    Rf_error("nodata %.17g does not fit %s", v, kName[d]);
  uint8_t buf[8] = {0};
  store(buf, 0, d, v);
  return sample_bits(buf, 0, d);
}

// Fails unless v can be stored at type d. Floats round to nearest.
static void check_value(int d, double v, R_xlen_t i, const char *what) {
  if (is_int_type(d)) {
    if (!int_fits(d, v))
      Rf_error("%s %.17g at position %.0f does not fit %s", what, v,
               (double)i + 1, kName[d]);
  } else if (d == GEOZL_DT_F32) {
    if (R_FINITE(v) && fabs(v) > FLT_MAX)
      Rf_error("%s %.17g at position %.0f is outside the float32 range", what,
               v, (double)i + 1);
  } else if (d == GEOZL_DT_F16) {
    int overflow;
    f64_to_f16(v, &overflow);
    if (overflow)
      Rf_error("%s %.17g at position %.0f is outside the float16 range", what,
               v, (double)i + 1);
  }
}

// R values to native samples. A raw vector is taken as samples already. NA
// and NaN become nodata when one is given; float types otherwise keep them,
// payload included, and an integer type has no way to hold them.
SEXP geozl_r_pack(SEXP x, SEXP dtype_s, SEXP nodata_s) {
  int d = dtype_arg(dtype_s);
  size_t w = kWidth[d];
  if (TYPEOF(x) == RAWSXP) {
    if ((size_t)XLENGTH(x) % w != 0)
      Rf_error("a raw vector of %s samples needs a length divisible by %d",
               kName[d], (int)w);
    return x;
  }
  if (TYPEOF(x) != INTSXP && TYPEOF(x) != REALSXP)
    Rf_error("`x` must be a double, integer or raw vector, not %s",
             Rf_type2char(TYPEOF(x)));
  int hasNodata = !Rf_isNull(nodata_s);
  double nodata = hasNodata ? Rf_asReal(nodata_s) : 0.0;
  R_xlen_t n = XLENGTH(x);
  SEXP out = PROTECT(Rf_allocVector(RAWSXP, n * (R_xlen_t)w));
  uint8_t *dst = RAW(out);
  const int *ix = TYPEOF(x) == INTSXP ? INTEGER(x) : NULL;
  const double *dx = ix ? NULL : REAL(x);
  for (R_xlen_t i = 0; i < n; ++i) {
    double v;
    int missing;
    if (ix) {
      missing = ix[i] == NA_INTEGER;
      v = missing ? NA_REAL : ix[i];
    } else {
      v = dx[i];
      missing = ISNAN(v);
    }
    if (missing) {
      if (hasNodata)
        v = nodata;
      else if (is_int_type(d))
        Rf_error("`x` has NA at position %.0f; an integer datatype needs "
                 "`nodata` to mark it",
                 (double)i + 1);
    }
    check_value(d, v, i, "value");
    store(dst, i, d, v);
  }
  UNPROTECT(1);
  return out;
}

// as: 0 raw, 1 double, 2 integer. Samples equal to nodata become NA.
SEXP geozl_r_unpack(SEXP bytes, SEXP dtype_s, SEXP as_s, SEXP nodata_s) {
  int d = dtype_arg(dtype_s);
  int as = Rf_asInteger(as_s);
  const uint8_t *src = raw_arg(bytes, "frame");
  size_t w = kWidth[d];
  if ((size_t)XLENGTH(bytes) % w != 0)
    Rf_error("%.0f bytes do not hold whole %s samples",
             (double)XLENGTH(bytes), kName[d]);
  if (as == 0)
    return bytes;
  R_xlen_t n = XLENGTH(bytes) / (R_xlen_t)w;
  int hasNodata = !Rf_isNull(nodata_s);
  double nodata = hasNodata ? Rf_asReal(nodata_s) : 0.0;
  uint64_t sentinel = hasNodata ? nodata_bits(d, nodata) : 0;
  if (as == 1) {
    SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
    double *o = REAL(out);
    R_xlen_t inexact = 0;
    for (R_xlen_t i = 0; i < n; ++i) {
      double v = load(src, i, d);
      if ((d == GEOZL_DT_U64 || d == GEOZL_DT_I64) && fabs(v) > 9007199254740992.0)
        ++inexact;
      o[i] = hasNodata && sample_bits(src, i, d) == sentinel ? NA_REAL : v;
    }
    if (inexact)
      Rf_warning("%.0f %s values are above 2^53 and lost precision as double",
                 (double)inexact, kName[d]);
    UNPROTECT(1);
    return out;
  }
  if (!is_int_type(d))
    Rf_error("%s samples need as = \"double\"", kName[d]);
  SEXP out = PROTECT(Rf_allocVector(INTSXP, n));
  int *o = INTEGER(out);
  for (R_xlen_t i = 0; i < n; ++i) {
    double v = load(src, i, d);
    if (hasNodata && sample_bits(src, i, d) == sentinel) {
      o[i] = NA_INTEGER;
      continue;
    }
    // INT_MIN is R's integer NA.
    if (v < -2147483647.0 || v > 2147483647.0)
      Rf_error("%s value %.17g at position %.0f does not fit an R integer; "
               "use as = \"double\"",
               kName[d], v, (double)i + 1);
    o[i] = (int)v;
  }
  UNPROTECT(1);
  return out;
}

// Whether float samples hold a NaN, which selects NaN nodata.
SEXP geozl_r_has_nan(SEXP bytes, SEXP dtype_s) {
  int d = dtype_arg(dtype_s);
  const uint8_t *src = raw_arg(bytes, "x");
  if (is_int_type(d))
    return Rf_ScalarLogical(0);
  R_xlen_t n = XLENGTH(bytes) / (R_xlen_t)kWidth[d];
  for (R_xlen_t i = 0; i < n; ++i)
    if (ISNAN(load(src, i, d)))
      return Rf_ScalarLogical(1);
  return Rf_ScalarLogical(0);
}

static void graph_finalizer(SEXP ptr) {
  geozl_2d_graph *g = R_ExternalPtrAddr(ptr);
  if (g != NULL) {
    geozl_2d_graph_close_c(g);
    R_ClearExternalPtr(ptr);
  }
}

static geozl_2d_graph *graph_of(SEXP ptr) {
  if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("geozl_graph"))
    Rf_error("`graph` must come from geozl_graph()");
  geozl_2d_graph *g = R_ExternalPtrAddr(ptr);
  if (g == NULL)
    Rf_error("this graph is no longer open; a saved graph does not survive a "
             "new session, build it again with geozl_graph()");
  return g;
}

SEXP geozl_r_graph_open(SEXP bytes, SEXP method_s, SEXP width_s, SEXP planes_s,
                        SEXP error_s, SEXP dtype_s, SEXP mode_s,
                        SEXP nodata_s) {
  int d = dtype_arg(dtype_s);
  const char *method = string_arg(method_s, "method");
  const char *recipe = error_arg(error_s);
  uint32_t width = u32_arg(width_s, "width");
  uint32_t planes = u32_arg(planes_s, "planes");
  int mode = Rf_asInteger(mode_s);
  uint64_t bits = mode == GEOZL_NODATA_VALUE ? nodata_bits(d, Rf_asReal(nodata_s)) : 0;
  const uint8_t *src = raw_arg(bytes, "x");
  size_t w = kWidth[d];
  char err[ERR_SIZE] = {0};
  geozl_2d_graph *g = NULL;
  int rc = geozl_2d_graph_open_c(&g, method, width, planes, recipe, d, mode,
                                 bits, src, (size_t)XLENGTH(bytes) / w, w, err,
                                 sizeof err);
  if (rc != 0)
    Rf_error("geozl_graph failed (method = \"%s\"): %s (ZL error code %d)",
             method, err, rc);
  SEXP ptr = PROTECT(R_MakeExternalPtr(g, Rf_install("geozl_graph"), R_NilValue));
  R_RegisterCFinalizerEx(ptr, graph_finalizer, TRUE);
  UNPROTECT(1);
  return ptr;
}

// A list of whole numbers in int32 range to a coefficient blob. Memory comes
// from R_alloc, so an error anywhere leaks nothing.
static uint8_t *pack_coeffs(SEXP list, size_t *size) {
  if (TYPEOF(list) != VECSXP)
    Rf_error("`coeffs` must be a list of integer vectors");
  R_xlen_t k = XLENGTH(list);
  if (k == 0)
    Rf_error("`coeffs` must contain at least one vector");
  if (k > GEOZL_COEFFS_MAX_VECS)
    Rf_error("`coeffs` supports at most %d vectors", GEOZL_COEFFS_MAX_VECS);
  uint32_t *counts = (uint32_t *)R_alloc((size_t)k, sizeof(uint32_t));
  const int32_t **vecs = (const int32_t **)R_alloc((size_t)k, sizeof(int32_t *));
  size_t used = 6;
  for (R_xlen_t j = 0; j < k; ++j) {
    SEXP v = VECTOR_ELT(list, j);
    if (TYPEOF(v) != INTSXP && TYPEOF(v) != REALSXP)
      Rf_error("coeffs vector %.0f must be integer or double", (double)j + 1);
    R_xlen_t n = XLENGTH(v);
    if (n == 0)
      Rf_error("coeffs vector %.0f is empty", (double)j + 1);
    used += 4 + 4 * (size_t)n;
    if (used > GEOZL_COEFFS_MAX_BYTES)
      Rf_error("coeffs exceeds the %d-byte limit", GEOZL_COEFFS_MAX_BYTES);
    int32_t *vals = (int32_t *)R_alloc((size_t)n, sizeof(int32_t));
    for (R_xlen_t i = 0; i < n; ++i) {
      double x = TYPEOF(v) == INTSXP
                     ? (INTEGER(v)[i] == NA_INTEGER ? NA_REAL : INTEGER(v)[i])
                     : REAL(v)[i];
      if (!int_fits(GEOZL_DT_I32, x))
        Rf_error("coefficient %.0f of vector %.0f is not a whole number in "
                 "int32 range",
                 (double)i + 1, (double)j + 1);
      vals[i] = (int32_t)x;
    }
    counts[j] = (uint32_t)n;
    vecs[j] = vals;
  }
  size_t need = geozl_coeffs_size(counts, (size_t)k);
  if (need == 0)
    Rf_error("coeffs has an invalid shape");
  uint8_t *blob = (uint8_t *)R_alloc(need, 1);
  char err[ERR_SIZE] = {0};
  *size = geozl_coeffs_pack(blob, need, vecs, counts, (size_t)k, err, sizeof err);
  if (*size == 0)
    Rf_error("%s", err[0] ? err : "coeffs could not be packed");
  return blob;
}

static SEXP shrink(SEXP raw, size_t size) {
  if ((size_t)XLENGTH(raw) == size)
    return raw;
  SEXP out = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)size));
  memcpy(RAW(out), RAW(raw), size);
  UNPROTECT(1);
  return out;
}

SEXP geozl_r_compress(SEXP ptr, SEXP bytes, SEXP coeffs_s, SEXP itemsize_s) {
  geozl_2d_graph *g = graph_of(ptr);
  const uint8_t *src = raw_arg(bytes, "x");
  size_t w = (size_t)Rf_asInteger(itemsize_s);
  size_t len = (size_t)XLENGTH(bytes), n = len / w;
  size_t blobSize = 0;
  const uint8_t *blob = Rf_isNull(coeffs_s) ? NULL : pack_coeffs(coeffs_s, &blobSize);
  // Past the worst case; an incompressible tile still fits in 1.5x.
  size_t cap = 1024 + len + len / 2 + blobSize;
  SEXP dst = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)cap));
  char err[ERR_SIZE] = {0};
  size_t outSize = 0;
  int rc = blob == NULL
               ? geozl_2d_compress_graph_c(g, src, n, RAW(dst), cap, &outSize,
                                           err, sizeof err)
               : geozl_2d_compress_coeffs_c(g, src, n, blob, blobSize, RAW(dst),
                                            cap, &outSize, err, sizeof err);
  if (rc != 0)
    Rf_error("geozl_compress failed: %s (ZL error code %d)", err, rc);
  SEXP out = shrink(dst, outSize);
  UNPROTECT(1);
  return out;
}

SEXP geozl_r_decompress(SEXP frame, SEXP verify_s, SEXP max_s) {
  const uint8_t *src = raw_arg(frame, "frame");
  size_t len = (size_t)XLENGTH(frame);
  size_t dsize = geozl_2d_frame_dsize_c(src, len);
  if (dsize == 0)
    Rf_error("geozl_decompress: unreadable frame");
  if (!Rf_isNull(max_s) && (double)dsize > Rf_asReal(max_s))
    Rf_error("geozl_decompress: the frame declares %.0f bytes of output, above "
             "the %.0f allowed",
             (double)dsize, Rf_asReal(max_s));
  SEXP out = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t)dsize));
  // Numeric output must be 8-byte aligned; R_alloc memory is aligned for double.
  uint8_t *dst = RAW(out);
  if ((uintptr_t)dst % 8 != 0)
    dst = (uint8_t *)R_alloc(dsize, 1);
  char err[ERR_SIZE] = {0};
  size_t outSize = 0;
  int rc = geozl_2d_decompress_c(src, len, dst, dsize, &outSize,
                                 Rf_asLogical(verify_s), err, sizeof err);
  if (rc != 0)
    Rf_error("geozl_decompress failed: %s (ZL error code %d)", err, rc);
  if (dst != RAW(out))
    memcpy(RAW(out), dst, outSize);
  SEXP result = shrink(out, outSize);
  UNPROTECT(1);
  return result;
}

SEXP geozl_r_grid(SEXP prior_s, SEXP width_s) {
  enum { STRIDE = 48, CAP = 128 };
  const char *prior = string_arg(prior_s, "prior");
  char *names = R_alloc(STRIDE * CAP, 1);
  memset(names, 0, STRIDE * CAP);
  size_t count = 0;
  if (geozl_2d_grid_c(prior, (size_t)Rf_asInteger(width_s), names, STRIDE, CAP,
                      &count) != 0)
    return R_NilValue;
  if (count > CAP)
    count = CAP;
  SEXP out = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)count));
  for (size_t i = 0; i < count; ++i)
    SET_STRING_ELT(out, (R_xlen_t)i, Rf_mkCharCE(names + i * STRIDE, CE_UTF8));
  UNPROTECT(1);
  return out;
}

// One recipe: c(frame bytes, encode seconds, decode seconds), or NULL when the
// recipe does not apply to this input.
SEXP geozl_r_bench(SEXP bytes, SEXP method_s, SEXP width_s, SEXP planes_s,
                   SEXP error_s, SEXP dtype_s, SEXP mode_s, SEXP nodata_s,
                   SEXP reps_s, SEXP verify_s) {
  int d = dtype_arg(dtype_s);
  const char *method = string_arg(method_s, "method");
  const char *recipe = error_arg(error_s);
  uint32_t width = u32_arg(width_s, "width");
  uint32_t planes = u32_arg(planes_s, "planes");
  int mode = Rf_asInteger(mode_s);
  uint64_t bits = mode == GEOZL_NODATA_VALUE ? nodata_bits(d, Rf_asReal(nodata_s)) : 0;
  const uint8_t *src = raw_arg(bytes, "x");
  size_t w = kWidth[d];
  size_t reps = reps_arg(reps_s);
  size_t comp = 0;
  double enc = 0, dec = 0;
  char err[ERR_SIZE] = {0};
  // Checksums on, so the size is the frame geozl_compress writes.
  R_CheckUserInterrupt();
  int rc = geozl_2d_bench_c(method, width, planes, recipe, d, mode, bits, src,
                            (size_t)XLENGTH(bytes) / w, w, reps, 1,
                            Rf_asLogical(verify_s), &comp, &enc, &dec, err,
                            sizeof err);
  R_CheckUserInterrupt();
  if (rc != 0)
    return R_NilValue;
  SEXP out = PROTECT(Rf_allocVector(REALSXP, 3));
  REAL(out)[0] = (double)comp;
  REAL(out)[1] = enc;
  REAL(out)[2] = dec;
  UNPROTECT(1);
  return out;
}

SEXP geozl_r_coeffs(SEXP frame) {
  const uint8_t *src = raw_arg(frame, "frame");
  size_t len = (size_t)XLENGTH(frame), size = 0;
  int rc = geozl_2d_frame_coeffs_c(src, len, NULL, 0, &size);
  if (rc < 0)
    return R_NilValue;
  if (rc != 0)
    Rf_error("geozl_coeffs: unreadable frame (ZL error code %d)", rc);
  uint8_t *blob = (uint8_t *)R_alloc(size > 0 ? size : 1, 1);
  rc = geozl_2d_frame_coeffs_c(src, len, blob, size, &size);
  if (rc != 0)
    Rf_error("geozl_coeffs: unreadable frame (ZL error code %d)", rc);
  size_t nv = 0, nvals = 0;
  rc = geozl_coeffs_parse(blob, size, NULL, 0, NULL, 0, &nv, &nvals);
  if (rc == 1)
    return R_NilValue;
  if (rc != 0)
    Rf_error("invalid geozl coefficient blob (code %d)", rc);
  uint32_t *counts = (uint32_t *)R_alloc(nv > 0 ? nv : 1, sizeof(uint32_t));
  int32_t *vals = (int32_t *)R_alloc(nvals > 0 ? nvals : 1, sizeof(int32_t));
  rc = geozl_coeffs_parse(blob, size, vals, nvals, counts, nv, NULL, NULL);
  if (rc != 0)
    Rf_error("invalid geozl coefficient blob (code %d)", rc);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)nv));
  size_t off = 0;
  for (size_t j = 0; j < nv; ++j) {
    SEXP v = Rf_allocVector(INTSXP, (R_xlen_t)counts[j]);
    SET_VECTOR_ELT(out, (R_xlen_t)j, v);
    for (uint32_t i = 0; i < counts[j]; ++i)
      INTEGER(v)[i] = vals[off + i];
    off += counts[j];
  }
  UNPROTECT(1);
  return out;
}

static const R_CallMethodDef methods[] = {
    {"geozl_r_pack", (DL_FUNC)&geozl_r_pack, 3},
    {"geozl_r_unpack", (DL_FUNC)&geozl_r_unpack, 4},
    {"geozl_r_has_nan", (DL_FUNC)&geozl_r_has_nan, 2},
    {"geozl_r_graph_open", (DL_FUNC)&geozl_r_graph_open, 8},
    {"geozl_r_compress", (DL_FUNC)&geozl_r_compress, 4},
    {"geozl_r_decompress", (DL_FUNC)&geozl_r_decompress, 3},
    {"geozl_r_grid", (DL_FUNC)&geozl_r_grid, 2},
    {"geozl_r_bench", (DL_FUNC)&geozl_r_bench, 10},
    {"geozl_r_coeffs", (DL_FUNC)&geozl_r_coeffs, 1},
    {NULL, NULL, 0}};

void R_init_geozl(DllInfo *dll) {
  R_registerRoutines(dll, NULL, methods, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
