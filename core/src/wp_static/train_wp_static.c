// Fit the wp_static kernel from a tile, for 1 and 2 byte samples. Ridge normal
// equations for the coefficients, then the shift that gives the lowest residual
// byte plane entropy. Never loses to planar.

#include "train_wp_static.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define WP_SHIFT_MAX 8
#define WP_MAX_CAND  (WP_SHIFT_MAX + 2)

static int solve4(double A[4][4], double b[4], double x[4])
{
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        double best = fabs(A[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            double v = fabs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-12)
            return 0;
        if (piv != col) {
            for (int k = 0; k < 4; ++k) { double t = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < 4; ++r) {
            double m = A[r][col] / A[col][col];
            for (int k = col; k < 4; ++k) A[r][k] -= m * A[col][k];
            b[r] -= m * b[col];
        }
    }
    for (int r = 3; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < 4; ++k) s -= A[r][k] * x[k];
        x[r] = s / A[r][r];
    }
    return 1;
}

#define WP_ACCUM(T)                                                          \
    do {                                                                     \
        const T* s = (const T*)src;                                          \
        for (size_t r = 2; r < rows; ++r) {                                  \
            const T* row = s + r * w;                                        \
            const T* up  = s + (r - 1) * w;                                  \
            const T* up2 = s + (r - 2) * w;                                  \
            for (size_t c = 1; c + 1 < w; ++c) {                             \
                double f[4] = { (double)up[c], (double)up[c - 1],            \
                                (double)up[c + 1], (double)up2[c] };         \
                double tgt = (double)row[c] - (double)row[c - 1];           \
                for (int i = 0; i < 4; ++i) {                               \
                    v[i] += f[i] * tgt;                                      \
                    for (int j = 0; j < 4; ++j) M[i][j] += f[i] * f[j];      \
                }                                                            \
            }                                                                \
        }                                                                    \
    } while (0)

static double plane_H(const uint32_t* hist, size_t nb)
{
    const double invN = 1.0 / (double)nb;
    double H = 0.0;
    for (int k = 0; k < 256; ++k) {
        uint32_t ct = hist[k];
        if (ct) { double p = ct * invN; H -= p * log2(p); }
    }
    return H;
}

// All candidates in one pass. The residual matches wp_static_encode, accumulated
// in int64, boundary neighbours zero.
#define WP_SCORE(T, ZCAST)                                                   \
    do {                                                                     \
        const T* s = (const T*)src;                                          \
        for (size_t r = 0; r < rows; ++r) {                                  \
            const size_t row = r * w;                                        \
            const T* ab  = (r >= 1) ? s + row - w : (const T*)0;             \
            const T* ab2 = (r >= 2) ? s + row - 2 * w : (const T*)0;         \
            for (size_t c = 0; c < w; ++c) {                                 \
                const int64_t N  = ab ? ab[c] : 0;                          \
                const int64_t NW = (ab && c > 0) ? ab[c - 1] : 0;           \
                const int64_t NE = (ab && c + 1 < w) ? ab[c + 1] : 0;       \
                const int64_t NN = ab2 ? ab2[c] : 0;                        \
                const int64_t base =                                         \
                    (int64_t)s[row + c] - (int64_t)((c > 0) ? s[row + c - 1] : 0); \
                for (int m = 0; m < nc; ++m) {                              \
                    const int64_t K = (cf[m][0] * N + cf[m][1] * NW         \
                                       + cf[m][2] * NE + cf[m][3] * NN       \
                                       + rnd[m]) >> sh[m];                   \
                    const int16_t z = (int16_t)(ZCAST)(base - K);           \
                    const uint16_t zz = (uint16_t)(((uint16_t)z << 1) ^ (z >> 15)); \
                    hi[m][(zz >> 8) & 0xFF]++;                              \
                    lo[m][zz & 0xFF]++;                                     \
                }                                                            \
            }                                                                \
        }                                                                    \
    } while (0)

void wp_static_train(int16_t coeffs[4], uint8_t* shift, const void* src,
                     size_t width, size_t nb_elts, size_t elt_width)
{
    coeffs[0] = 1; coeffs[1] = -1; coeffs[2] = 0; coeffs[3] = 0;
    *shift = 0;

    if (nb_elts == 0 || (elt_width != 1 && elt_width != 2))
        return;
    const size_t w = (width == 0 || width > nb_elts) ? nb_elts : width;
    const size_t rows = nb_elts / w;
    if (rows < 3 || w < 3)
        return;

    double M[4][4] = {{0}}, v[4] = {0};
    if (elt_width == 1) WP_ACCUM(uint8_t); else WP_ACCUM(uint16_t);

    double tr = M[0][0] + M[1][1] + M[2][2] + M[3][3];
    double lambda = 1e-6 * (tr / 4.0) + 1e-9; // ridge, the neighbours are collinear
    for (int i = 0; i < 4; ++i) M[i][i] += lambda;

    double a[4];
    if (!solve4(M, v, a))
        return;

    int16_t cf[WP_MAX_CAND][4];
    uint8_t sh[WP_MAX_CAND];
    int32_t rnd[WP_MAX_CAND];
    int nc = 0;
    cf[0][0] = 1; cf[0][1] = -1; cf[0][2] = 0; cf[0][3] = 0; // planar
    sh[0] = 0; rnd[0] = 0; nc = 1;
    for (int s = 0; s <= WP_SHIFT_MAX; ++s) {
        int16_t q[4];
        int ok = 1;
        for (int k = 0; k < 4; ++k) {
            long qi = lround(a[k] * (double)(1 << s));
            if (qi < -32768 || qi > 32767) { ok = 0; break; }
            q[k] = (int16_t)qi;
        }
        if (!ok) continue;
        cf[nc][0] = q[0]; cf[nc][1] = q[1]; cf[nc][2] = q[2]; cf[nc][3] = q[3];
        sh[nc] = (uint8_t)s;
        rnd[nc] = s ? (int32_t)1 << (s - 1) : 0;
        ++nc;
    }

    uint32_t hi[WP_MAX_CAND][256], lo[WP_MAX_CAND][256];
    memset(hi, 0, sizeof hi);
    memset(lo, 0, sizeof lo);
    if (elt_width == 1) WP_SCORE(uint8_t, int8_t); else WP_SCORE(uint16_t, int16_t);

    int best = 0;
    double best_cost = plane_H(hi[0], nb_elts) + plane_H(lo[0], nb_elts);
    for (int m = 1; m < nc; ++m) {
        double e = plane_H(hi[m], nb_elts) + plane_H(lo[m], nb_elts);
        if (e < best_cost) { best_cost = e; best = m; }
    }

    coeffs[0] = cf[best][0]; coeffs[1] = cf[best][1];
    coeffs[2] = cf[best][2]; coeffs[3] = cf[best][3];
    *shift = sh[best];
}