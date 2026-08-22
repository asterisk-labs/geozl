// Tiles for the quant_log tests, generated from a seed so a run reproduces
// anywhere. Each one is a shape a relative bound is actually asked for, plus the
// ones the grid is most likely to break on.

#ifndef GEOZL_TEST_QUANT_LOG_TILES_H
#define GEOZL_TEST_QUANT_LOG_TILES_H

#include "quant_log/quant_log_dtype.h"
#include "quant_log/quant_log_half.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QT_W 128
#define QT_H 128
#define QT_N ((size_t)QT_W * QT_H)

typedef struct {
  const char *name;
  int dtype;
  size_t n;
  void *data;
} qt_tile;

static uint64_t qt_seed = 1;

static inline uint64_t qt_bits(void) {
  uint64_t x = qt_seed;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  qt_seed = x;
  return x;
}

static inline double qt_uni(void) {
  return (double)(qt_bits() >> 11) * (1.0 / 9007199254740992.0);
}

// Box-Muller. The tails are the point of a log grid, so a generator that clipped
// them would test nothing.
static inline double qt_norm(void) {
  double u = qt_uni();
  if (u < 1e-300)
    u = 1e-300;
  return sqrt(-2.0 * log(u)) * cos(6.283185307179586477 * qt_uni());
}

static inline void qt_put(void *p, int dtype, size_t i, double v) {
  switch (dtype) {
  case QLOG_U8:
    ((uint8_t *)p)[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    break;
  case QLOG_U16:
    ((uint16_t *)p)[i] = (uint16_t)(v < 0 ? 0 : (v > 65535 ? 65535 : v));
    break;
  case QLOG_U32:
    ((uint32_t *)p)[i] = (uint32_t)(v < 0 ? 0 : v);
    break;
  case QLOG_I16:
    ((int16_t *)p)[i] = (int16_t)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
    break;
  case QLOG_I32:
    ((int32_t *)p)[i] = (int32_t)v;
    break;
  case QLOG_F16:
    ((uint16_t *)p)[i] = quant_log_float_to_half((float)v);
    break;
  case QLOG_F32:
    ((float *)p)[i] = (float)v;
    break;
  default:
    ((double *)p)[i] = v;
    break;
  }
}

static inline double qt_get(const void *p, int dtype, size_t i) {
  switch (dtype) {
  case QLOG_U8:
    return ((const uint8_t *)p)[i];
  case QLOG_U16:
    return ((const uint16_t *)p)[i];
  case QLOG_U32:
    return ((const uint32_t *)p)[i];
  case QLOG_I16:
    return ((const int16_t *)p)[i];
  case QLOG_I32:
    return ((const int32_t *)p)[i];
  case QLOG_F16:
    return quant_log_half_to_float(((const uint16_t *)p)[i]);
  case QLOG_F32:
    return ((const float *)p)[i];
  default:
    return ((const double *)p)[i];
  }
}

// A smooth patch, so the tiles look like rasters rather than independent draws.
static inline void qt_smooth(double *f, int passes) {
  double *t = (double *)malloc(QT_N * sizeof(double));
  for (int p = 0; p < passes; ++p) {
    for (size_t y = 0; y < QT_H; ++y)
      for (size_t x = 0; x < QT_W; ++x) {
        const size_t xm = x ? x - 1 : x, xp = x + 1 < QT_W ? x + 1 : x;
        const size_t ym = y ? y - 1 : y, yp = y + 1 < QT_H ? y + 1 : y;
        t[y * QT_W + x] = 0.2 * (f[y * QT_W + x] + f[y * QT_W + xm] +
                                 f[y * QT_W + xp] + f[ym * QT_W + x] +
                                 f[yp * QT_W + x]);
      }
    memcpy(f, t, QT_N * sizeof(double));
  }
  free(t);
}

static inline qt_tile qt_make(const char *name) {
  qt_tile t;
  memset(&t, 0, sizeof(t));
  t.name = name;
  t.n = QT_N;
  t.dtype = QLOG_F32;
  qt_seed = 0xC0FFEEull ^ ((uint64_t)(unsigned char)name[0] * 1099511628211ull);

  double *f = (double *)malloc(QT_N * sizeof(double));
  for (size_t i = 0; i < QT_N; ++i)
    f[i] = qt_norm();
  qt_smooth(f, 3);

  if (strcmp(name, "counts_u16") == 0) {
    // Unsigned optical counts with a nodata
    // border of zeros.
    t.dtype = QLOG_U16;
    t.data = malloc(QT_N * 2);
    for (size_t y = 0; y < QT_H; ++y)
      for (size_t x = 0; x < QT_W; ++x) {
        const size_t i = y * QT_W + x;
        double v = 1500.0 * exp(1.1 * f[i]);
        if (x < 6 || y < 6 || x + 6 >= QT_W || y + 6 >= QT_H)
          v = 0.0;
        qt_put(t.data, t.dtype, i, v);
      }
  } else if (strcmp(name, "byte") == 0) {
    t.dtype = QLOG_U8;
    t.data = malloc(QT_N);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, 60.0 * exp(0.8 * f[i]));
  } else if (strcmp(name, "dem_i16") == 0) {
    // Elevation in metres, signed, crossing sea level.
    t.dtype = QLOG_I16;
    t.data = malloc(QT_N * 2);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, 400.0 * f[i]);
  } else if (strcmp(name, "counts_u32") == 0) {
    t.dtype = QLOG_U32;
    t.data = malloc(QT_N * 4);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, exp(11.0 * qt_uni()));
  } else if (strcmp(name, "reflectance") == 0) {
    t.data = malloc(QT_N * 4);
    for (size_t y = 0; y < QT_H; ++y)
      for (size_t x = 0; x < QT_W; ++x) {
        const size_t i = y * QT_W + x;
        double v = 0.15 * exp(1.4 * f[i]);
        if (v > 1.0)
          v = 1.0;
        if (x < 6 || y < 6 || x + 6 >= QT_W || y + 6 >= QT_H)
          v = 0.0;
        qt_put(t.data, t.dtype, i, v);
      }
  } else if (strcmp(name, "humidity") == 0) {
    t.data = malloc(QT_N * 4);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, 2.0e-4 * exp(2.6 * f[i]));
  } else if (strcmp(name, "precipitation") == 0) {
    t.data = malloc(QT_N * 4);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i,
             f[i] < 0.5 ? 0.0 : 1.0e-5 * exp(4.0 * (f[i] - 0.5)));
  } else if (strcmp(name, "kelvin") == 0) {
    // A narrow band away from zero, which is where STORE=VALUES applies.
    t.data = malloc(QT_N * 4);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, 290.0 + 14.0 * f[i]);
  } else if (strcmp(name, "utm") == 0) {
    // Easting and northing in metres, far from zero and needing more digits than
    // a float carries.
    t.dtype = QLOG_F64;
    t.data = malloc(QT_N * 8);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, 500000.0 + 40000.0 * f[i]);
  } else if (strcmp(name, "anomaly") == 0) {
    t.data = malloc(QT_N * 4);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, f[i] * exp(3.0 * qt_norm()));
  } else if (strcmp(name, "wide") == 0) {
    t.dtype = QLOG_F64;
    t.data = malloc(QT_N * 8);
    for (size_t i = 0; i < QT_N; ++i)
      qt_put(t.data, t.dtype, i, exp(690.0 * f[i] * 0.45));
  } else if (strcmp(name, "half") == 0) {
    t.dtype = QLOG_F16;
    t.data = malloc(QT_N * 2);
    for (size_t i = 0; i < QT_N; ++i) {
      double v = 0.15 * exp(1.4 * f[i]);
      qt_put(t.data, t.dtype, i, v > 1.0 ? 1.0 : v);
    }
  } else if (strcmp(name, "edges") == 0) {
    // Assembled, not drawn: every exponent of the type, both signs, the
    // subnormals, the ends, and the values that are not numbers.
    t.data = malloc(QT_N * 4);
    float *o = (float *)t.data;
    size_t k = 0;
    for (int e = 0; e < 255 && k + 2 < QT_N; ++e) {
      const uint32_t b = ((uint32_t)e << 23) | 0x123456u;
      float v;
      memcpy(&v, &b, sizeof(v));
      o[k++] = v;
      o[k++] = -v;
    }
    const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                           0x007FFFFFu, 0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu,
                           0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x3F800000u};
    for (size_t j = 0; j < sizeof sp / sizeof *sp && k < QT_N; ++j)
      memcpy(&o[k++], &sp[j], sizeof(float));
    while (k < QT_N) {
      o[k] = o[k % 128];
      ++k;
    }
  }
  free(f);
  return t;
}

static inline void qt_free(qt_tile *t) {
  free(t->data);
  t->data = NULL;
}

static const char *const qt_int_tiles[] = {"byte", "counts_u16", "dem_i16",
                                           "counts_u32"};
static const char *const qt_flt_tiles[] = {"reflectance", "humidity", "kelvin",
                                           "precipitation", "anomaly", "wide",
                                           "half",         "utm",     "edges"};
#define QT_INT_N (sizeof qt_int_tiles / sizeof *qt_int_tiles)
#define QT_FLT_N (sizeof qt_flt_tiles / sizeof *qt_flt_tiles)

#endif // GEOZL_TEST_QUANT_LOG_TILES_H
