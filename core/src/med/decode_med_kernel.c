// Inverse MED predictor, JPEG-LS (ITU-T T.87 A.4.2,
// https://www.itu.int/rec/T-REC-T.87/en).
//
// W comes from the row being reconstructed, so like average this is an IIR and
// cannot vectorize. Four rows run a column apart instead, which gives four
// independent chains and leaves the lower three reading N and NW from
// registers. The median is masked because the three way select mispredicts.

#include "decode_med_kernel.h"

#include "common/raster.h" // geozl_row_width

#include <stdint.h>

// Above four, the row streams start colliding in cache on power of two strides.
#define MED_ROWS 4

// ge is all ones when NW is at or above max(W, N), le when at or below the min.
#define MED_PREDICT(T, P, Wv, Nv, NWv)                                         \
  const T mn = (Wv) < (Nv) ? (Wv) : (Nv);                                      \
  const T mx = (Wv) < (Nv) ? (Nv) : (Wv);                                      \
  const T ge = (T)(0 - (T)((NWv) >= mx));                                      \
  const T le = (T)(0 - (T)((NWv) <= mn));                                      \
  const T P = (T)((T)(ge & mn) |                                               \
                  (T)((T)~ge & (T)((T)(le & mx) |                              \
                                   (T)((T)~le & (T)((Wv) + (Nv) - (NWv))))));

// The top row reads N from the row above and keeps it as the next NW.
#define MED_TOP(T, cc)                                                         \
  do {                                                                         \
    const size_t c = (cc);                                                     \
    MED_PREDICT(T, P, W0, up[c], nw)                                           \
    const T v = (T)(s0[c] + P);                                                \
    nw = up[c];                                                                \
    P0 = W0;                                                                   \
    W0 = v;                                                                    \
    d0[c] = v;                                                                 \
  } while (0)

#define MED_STEP(T, dr, sr, cc, Wv, Pv, Nv, NWv)                               \
  do {                                                                         \
    const size_t c = (cc);                                                     \
    MED_PREDICT(T, P, Wv, Nv, NWv)                                             \
    const T v = (T)((sr)[c] + P);                                              \
    (Pv) = (Wv);                                                               \
    (Wv) = v;                                                                  \
    (dr)[c] = v;                                                               \
  } while (0)

#define MED_DEC(T, B)                                                          \
  do {                                                                         \
    T *d = (T *)dst;                                                           \
    const T *s = (const T *)src;                                               \
    const size_t rows = nbElts / w;                                            \
    d[0] = s[0];                                                               \
    for (size_t c = 1; c < w; ++c)                                             \
      d[c] = (T)(s[c] + d[c - 1]);                                             \
    size_t r = 1;                                                              \
    /* A block needs the wavefront to fill and drain inside one row, so */     \
    /* narrow rows and short tiles take the serial path whole. */              \
    if (w >= 2 * MED_ROWS && rows > MED_ROWS) {                                \
      for (; r + MED_ROWS <= rows; r += MED_ROWS) {                            \
        T *d0 = d + r * w, *d1 = d0 + w, *d2 = d1 + w, *d3 = d2 + w;           \
        const T *s0 = s + r * w, *s1 = s0 + w, *s2 = s1 + w, *s3 = s2 + w;     \
        const T *up = d + (r - 1) * w;                                         \
        T nw = up[0];                                                          \
        d0[0] = (T)(s0[0] + up[0]);                                            \
        d1[0] = (T)(s1[0] + d0[0]);                                            \
        d2[0] = (T)(s2[0] + d1[0]);                                            \
        d3[0] = (T)(s3[0] + d2[0]);                                            \
        T W0 = d0[0], W1 = d1[0], W2 = d2[0], W3 = d3[0];                      \
        T P0 = 0, P1 = 0, P2 = 0, P3 = 0;                                      \
        (void)P3; /* nothing below the last row reads its predecessor */       \
        MED_TOP(T, 1);                                                         \
        MED_STEP(T, d1, s1, 1, W1, P1, W0, P0);                                \
        MED_TOP(T, 2);                                                         \
        MED_STEP(T, d2, s2, 1, W2, P2, W1, P1);                                \
        MED_STEP(T, d1, s1, 2, W1, P1, W0, P0);                                \
        MED_TOP(T, 3);                                                         \
        for (size_t t = MED_ROWS; t < w; ++t) {                                \
          MED_STEP(T, d3, s3, t - 3, W3, P3, W2, P2);                          \
          MED_STEP(T, d2, s2, t - 2, W2, P2, W1, P1);                          \
          MED_STEP(T, d1, s1, t - 1, W1, P1, W0, P0);                          \
          MED_TOP(T, t);                                                       \
        }                                                                      \
        MED_STEP(T, d3, s3, w - 3, W3, P3, W2, P2);                            \
        MED_STEP(T, d2, s2, w - 2, W2, P2, W1, P1);                            \
        MED_STEP(T, d1, s1, w - 1, W1, P1, W0, P0);                            \
        MED_STEP(T, d3, s3, w - 2, W3, P3, W2, P2);                            \
        MED_STEP(T, d2, s2, w - 1, W2, P2, W1, P1);                            \
        MED_STEP(T, d3, s3, w - 1, W3, P3, W2, P2);                            \
      }                                                                        \
    }                                                                          \
    for (; r < rows; ++r) {                                                    \
      const size_t row = r * w;                                                \
      d[row] = (T)(s[row] + d[row - w]);                                       \
      for (size_t c = 1; c < w; ++c) {                                         \
        const size_t i = row + c;                                              \
        MED_PREDICT(T, P, d[i - 1], d[i - w], d[i - w - 1])                    \
        d[i] = (T)(s[i] + P);                                                  \
      }                                                                        \
    }                                                                          \
  } while (0)

int med_decode(void *dst, const void *src, size_t width, size_t nbElts,
               size_t eltWidth) {
  GEOZL_ROW_DISPATCH(w, MED_DEC);
}

#undef MED_DEC
#undef MED_STEP
#undef MED_TOP
#undef MED_PREDICT
#undef MED_ROWS
