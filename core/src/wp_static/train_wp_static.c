// Fit the wp_static kernel from a tile. The kernel rides in the codec header, so
// this only picks coefficients and its one rule is to never lose to planar. Ridge
// normal equations, then a shift sweep scored by the byte plane entropy of the
// residual, the way the entropy stage codes it. Defined for 1 and 2 byte samples.

#include "train_wp_static.h"
#include "encode_wp_static_kernel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define WP_SHIFT_MAX 8

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

// Cost of the residual as the entropy stage codes it, the summed order zero
// entropy of the high and low byte planes of the zigzagged residual.
static double cand_cost(void* scratch, const void* src, size_t w, size_t nb,
                        size_t elt, const int16_t c[4], uint8_t shift,
                        uint32_t* hi, uint32_t* lo)
{
    wp_static_encode(scratch, src, w, nb, elt, c, shift);
    memset(hi, 0, 256 * sizeof(uint32_t));
    memset(lo, 0, 256 * sizeof(uint32_t));
    const uint16_t* r = (const uint16_t*)scratch;
    const uint8_t* r8 = (const uint8_t*)scratch;
    for (size_t i = 0; i < nb; ++i) {
        int16_t s = (elt == 1) ? (int16_t)(int8_t)r8[i] : (int16_t)r[i];
        uint16_t zz = (uint16_t)((s << 1) ^ (s >> 15));
        hi[(zz >> 8) & 0xFF]++;
        lo[zz & 0xFF]++;
    }
    return plane_H(hi, nb) + plane_H(lo, nb);
}

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
    double lambda = 1e-6 * (tr / 4.0) + 1e-9;         // ridge, the neighbours are collinear
    for (int i = 0; i < 4; ++i) M[i][i] += lambda;

    double a[4];
    if (!solve4(M, v, a))
        return;

    void* scratch = malloc(nb_elts * elt_width);
    if (!scratch) return;
    uint32_t* hi = (uint32_t*)malloc(256 * sizeof(uint32_t));
    uint32_t* lo = (uint32_t*)malloc(256 * sizeof(uint32_t));
    if (!hi || !lo) { free(scratch); free(hi); free(lo); return; }

    const int16_t planar[4] = {1, -1, 0, 0};
    double best = cand_cost(scratch, src, w, nb_elts, elt_width, planar, 0, hi, lo);

    for (int sh = 0; sh <= WP_SHIFT_MAX; ++sh) {
        int16_t q[4];
        int ok = 1;
        for (int k = 0; k < 4; ++k) {
            long qi = lround(a[k] * (double)(1 << sh));
            if (qi < -32768 || qi > 32767) { ok = 0; break; }
            q[k] = (int16_t)qi;
        }
        if (!ok) continue;
        double e = cand_cost(scratch, src, w, nb_elts, elt_width, q, (uint8_t)sh, hi, lo);
        if (e < best) {
            best = e;
            coeffs[0] = q[0]; coeffs[1] = q[1]; coeffs[2] = q[2]; coeffs[3] = q[3];
            *shift = (uint8_t)sh;
        }
    }
    free(hi); free(lo); free(scratch);
}