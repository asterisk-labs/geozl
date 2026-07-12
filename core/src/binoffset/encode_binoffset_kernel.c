#include "encode_binoffset_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// v|1 keeps clz defined at 0; the mask forces bl = 0 for v == 0
#if defined(__GNUC__) || defined(__clang__)
static inline unsigned bo_bitlen32(uint32_t v) {
  return (32u - (unsigned)__builtin_clz(v | 1u)) & (unsigned)-(v != 0u);
}
static inline unsigned bo_bitlen64(uint64_t v) {
  return (64u - (unsigned)__builtin_clzll(v | 1u)) & (unsigned)-(v != 0u);
}
#else
static inline unsigned bo_bitlen32(uint32_t v) {
  unsigned n = 0;
  while (v) {
    v >>= 1;
    ++n;
  }
  return n;
}
static inline unsigned bo_bitlen64(uint64_t v) {
  unsigned n = 0;
  while (v) {
    v >>= 1;
    ++n;
  }
  return n;
}
#endif

#define BINOFFSET_SPLIT(T, TBITS, BITLEN)                                      \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *o = (T *)offsets;                                                       \
    for (size_t k = 0; k < nb_elts; ++k) {                                     \
      const T v = s[k];                                                        \
      const unsigned bl = BITLEN(v);                                           \
      const T msb = (T)((T)1 << ((bl - 1u) & (TBITS - 1u)));                   \
      bins[k] = (uint8_t)bl;                                                   \
      o[k] = (T)(v & (T)(msb - 1u));                                           \
    }                                                                          \
  } while (0)

void binoffset_split(uint8_t *bins, void *offsets, const void *src,
                     size_t nb_elts, size_t elt_width) {
  switch (elt_width) {
  case 1:
    BINOFFSET_SPLIT(uint8_t, 8u, bo_bitlen32);
    break;
  case 2:
    BINOFFSET_SPLIT(uint16_t, 16u, bo_bitlen32);
    break;
  case 4:
    BINOFFSET_SPLIT(uint32_t, 32u, bo_bitlen32);
    break;
  case 8:
    BINOFFSET_SPLIT(uint64_t, 64u, bo_bitlen64);
    break;
  default:
    break;
  }
}

#define BINOFFSET_SPLIT_TABLE(T)                                               \
  do {                                                                         \
    const T *s = (const T *)src;                                               \
    T *o = (T *)offsets;                                                       \
    for (size_t k = 0; k < nb_elts; ++k) {                                     \
      const T v = s[k];                                                        \
      unsigned lo = 0, n = nb_bins;                                            \
      while (n > 1) {                                                          \
        const unsigned half = n / 2;                                           \
        lo += ((uint64_t)v >= lowers[lo + half]) ? half : 0;                   \
        n -= half;                                                             \
      }                                                                        \
      bins[k] = (uint8_t)lo;                                                   \
      o[k] = (T)(v - (T)lowers[lo]);                                           \
    }                                                                          \
  } while (0)

void binoffset_split_table(uint8_t *bins, void *offsets, const void *src,
                           size_t nb_elts, size_t elt_width,
                           const uint64_t lowers[256],
                           const uint8_t offset_bits[256], unsigned nb_bins) {
  (void)offset_bits; // bins already sized by the table builder
  switch (elt_width) {
  case 1:
    BINOFFSET_SPLIT_TABLE(uint8_t);
    break;
  case 2:
    BINOFFSET_SPLIT_TABLE(uint16_t);
    break;
  case 4:
    BINOFFSET_SPLIT_TABLE(uint32_t);
    break;
  case 8:
    BINOFFSET_SPLIT_TABLE(uint64_t);
    break;
  default:
    break;
  }
}

// pcodec histograms.rs (apply_sorted path)

typedef struct {
  uint64_t n;
  uint64_t n_bins;
  unsigned n_bins_log;
  size_t n_applied;
  size_t next_avail_bin_idx;
  int has_incomplete;
  uint64_t inc_lower;
  uint64_t inc_upper;
  size_t inc_count;
  uint64_t *out_lowers;
  uint64_t *out_uppers;
  uint32_t *out_counts;
  size_t n_out;
} bo_hist;

static inline size_t bo_bin_idx(const bo_hist *h, size_t c_count) {
  return (size_t)(((uint64_t)c_count << h->n_bins_log) / h->n);
}
static inline size_t bo_c_count(const bo_hist *h, size_t bin_idx) {
  return (size_t)((((uint64_t)bin_idx + 1) * h->n + h->n_bins - 1) >> h->n_bins_log);
}

static void bo_apply_incomplete(bo_hist *h, uint64_t lower, uint64_t upper,
                                size_t len) {
  if (len == 0)
    return;
  if (h->has_incomplete) {
    h->inc_upper = upper;
    h->inc_count += len;
  } else {
    h->inc_lower = lower;
    h->inc_upper = upper;
    h->inc_count = len;
    h->has_incomplete = 1;
  }
  h->n_applied += len;
}

static int bo_complete_bin(bo_hist *h, size_t bin_idx) {
  if (!h->has_incomplete)
    return 0;
  h->next_avail_bin_idx = bin_idx + 1;
  h->out_lowers[h->n_out] = h->inc_lower;
  h->out_uppers[h->n_out] = h->inc_upper;
  h->out_counts[h->n_out] = (uint32_t)h->inc_count;
  h->n_out++;
  h->has_incomplete = 0;
  return 1;
}

static void bo_apply_constant_run(bo_hist *h, uint64_t v, size_t len) {
  const size_t start = h->n_applied;
  const size_t mid = start + len / 2;
  const size_t end = start + len;
  size_t bin_idx = bo_bin_idx(h, mid);
  if (bin_idx > h->next_avail_bin_idx) {
    const size_t spare = bin_idx - 1;
    if (!bo_complete_bin(h, spare))
      bin_idx = spare;
  }
  bo_apply_incomplete(h, v, v, len);
  if (end >= bo_c_count(h, bin_idx))
    bo_complete_bin(h, bin_idx);
}

static void bo_apply_sorted(bo_hist *h, const uint64_t *lat, size_t len) {
  size_t off = 0;
  while (off < len) {
    const uint64_t *cur = lat + off;
    const size_t rem = len - off;
    const size_t target_bin_idx = bo_bin_idx(h, h->n_applied);
    const size_t target_c_count = bo_c_count(h, target_bin_idx);
    const size_t target_i = target_c_count - h->n_applied;

    if (target_i >= rem) {
      bo_apply_incomplete(h, cur[0], cur[rem - 1], rem);
      if (target_i == rem)
        bo_complete_bin(h, target_bin_idx);
      break;
    }

    size_t l = target_i - 1;
    size_t r = target_i;
    const uint64_t target_x = cur[l];
    while (l > 0 && cur[l - 1] == target_x)
      l--;
    while (r < rem && cur[r] == target_x)
      r++;

    if (l > 0)
      bo_apply_incomplete(h, cur[0], cur[l - 1], l);
    bo_apply_constant_run(h, target_x, r - l);
    off += r;
  }
}

size_t binoffset_histogram(const uint64_t *sorted, size_t n, unsigned n_bins_log,
                           uint64_t *out_lowers, uint64_t *out_uppers,
                           uint32_t *out_counts) {
  if (n == 0)
    return 0;
  bo_hist h;
  h.n = (uint64_t)n;
  h.n_bins = (uint64_t)1 << n_bins_log;
  h.n_bins_log = n_bins_log;
  h.n_applied = 0;
  h.next_avail_bin_idx = 0;
  h.has_incomplete = 0;
  h.inc_lower = 0;
  h.inc_upper = 0;
  h.inc_count = 0;
  h.out_lowers = out_lowers;
  h.out_uppers = out_uppers;
  h.out_counts = out_counts;
  h.n_out = 0;
  bo_apply_sorted(&h, sorted, n);
  return h.n_out;
}

// pcodec bin_optimization.rs log2_approx (f32, to match its rounding)
static inline float bo_log2_approx(float x) {
  const float Z = 0.674f;
  const uint32_t SIGNIF_MASK = 0x7FFFFFu;
  uint32_t z_bits;
  memcpy(&z_bits, &Z, 4);
  const uint32_t Z_SIGNIF = z_bits & SIGNIF_MASK;
  const float B = 2.0f / Z;
  const float C = -B / (6.0f * Z);
  const float A = -B - C;

  uint32_t bits;
  memcpy(&bits, &x, 4);
  const uint32_t exp = bits >> 23;
  const uint32_t signif = bits & SIGNIF_MASK;
  const uint32_t high_bit = (signif > Z_SIGNIF) ? 1u : 0u;
  const uint32_t log_int = exp + high_bit - 127u;

  const uint32_t e2 = 0x7Fu ^ high_bit;
  const uint32_t nbits = (e2 << 23) | signif;
  float normalized;
  memcpy(&normalized, &nbits, 4);
  return (float)log_int + A + normalized * (B + C * normalized);
}

static inline float bo_bin_cost(float meta_bits, uint64_t lower, uint64_t upper,
                                uint32_t count, float total_log2) {
  const float c = (float)count;
  const float ans_cost = total_log2 - bo_log2_approx(c);
  const float offset_cost = (float)bo_bitlen64(upper - lower);
  return meta_bits + (ans_cost + offset_cost) * c;
}

#define BO_SPEEDUP_WORTH_BITS_PER_NUM 0.1f

unsigned binoffset_optimize_table(const uint64_t *fine_lowers,
                                  const uint64_t *fine_uppers,
                                  const uint32_t *fine_counts, size_t n_fine,
                                  double meta_bits, uint64_t *out_lowers,
                                  uint8_t *out_offset_bits, unsigned max_bins) {
  if (n_fine == 0 || max_bins == 0)
    return 0;
  const size_t B = n_fine;
  const float meta_f = (float)meta_bits;

  uint64_t *csum = (uint64_t *)malloc((B + 1) * sizeof(uint64_t));
  float *best = (float *)malloc((B + 1) * sizeof(float));
  size_t *best_j = (size_t *)malloc((B + 1) * sizeof(size_t));
  size_t *starts = (size_t *)malloc(B * sizeof(size_t));
  if (!csum || !best || !best_j || !starts) {
    free(csum);
    free(best);
    free(best_j);
    free(starts);
    return 0;
  }

  csum[0] = 0;
  for (size_t i = 0; i < B; ++i)
    csum[i + 1] = csum[i] + (uint64_t)fine_counts[i];
  const uint64_t total = csum[B];
  if (total == 0) {
    free(csum);
    free(best);
    free(best_j);
    free(starts);
    return 0;
  }
  const float log2_total = bo_log2_approx((float)total);

  best[0] = 0.0f;
  best_j[0] = 0;
  for (size_t i = 1; i <= B; ++i) {
    const uint64_t hi = fine_uppers[i - 1];
    float bi = INFINITY;
    size_t bj = 0;
    for (size_t j = i; j-- > 0;) {
      const uint32_t cnt = (uint32_t)(csum[i] - csum[j]);
      const float cost =
          best[j] + bo_bin_cost(meta_f, fine_lowers[j], hi, cnt, log2_total);
      if (cost < bi) {
        bi = cost;
        bj = j;
      }
    }
    best[i] = bi;
    best_j[i] = bj;
  }
  const float best_cost = best[B];
  const float total_f = (float)total;

  const float single_cost =
      bo_bin_cost(meta_f, fine_lowers[0], fine_uppers[B - 1],
                  (uint32_t)total, log2_total);
  if (single_cost < best_cost + BO_SPEEDUP_WORTH_BITS_PER_NUM * total_f) {
    out_lowers[0] = fine_lowers[0];
    out_offset_bits[0] =
        (uint8_t)bo_bitlen64(fine_uppers[B - 1] - fine_lowers[0]);
    free(csum);
    free(best);
    free(best_j);
    free(starts);
    return 1;
  }

  int all_trivial = 1;
  for (size_t k = 0; k < B; ++k)
    if (fine_lowers[k] != fine_uppers[k]) {
      all_trivial = 0;
      break;
    }
  if (all_trivial) {
    float tcost = 0.0f;
    for (size_t k = 0; k < B; ++k)
      tcost += bo_bin_cost(meta_f, fine_lowers[k], fine_uppers[k],
                           fine_counts[k], log2_total);
    if (tcost < best_cost + BO_SPEEDUP_WORTH_BITS_PER_NUM * total_f) {
      const unsigned nb = (B > (size_t)max_bins) ? max_bins : (unsigned)B;
      for (unsigned g = 0; g < nb; ++g) {
        const size_t end = (g + 1u < nb) ? (size_t)g : (B - 1);
        out_lowers[g] = fine_lowers[g];
        out_offset_bits[g] =
            (uint8_t)bo_bitlen64(fine_uppers[end] - fine_lowers[g]);
      }
      free(csum);
      free(best);
      free(best_j);
      free(starts);
      return nb;
    }
  }

  size_t ng = 0;
  for (size_t i = B; i > 0;) {
    const size_t j = best_j[i];
    starts[ng++] = j;
    i = j;
  }
  // starts is descending: ascending group g is starts[ng - 1 - g]
  const unsigned nb = (ng > (size_t)max_bins) ? max_bins : (unsigned)ng;
  for (unsigned g = 0; g < nb; ++g) {
    const size_t start = starts[ng - 1 - g];
    const size_t end =
        (g + 1u < nb) ? (starts[ng - 1 - (g + 1)] - 1) : (B - 1);
    out_lowers[g] = fine_lowers[start];
    out_offset_bits[g] = (uint8_t)bo_bitlen64(fine_uppers[end] - fine_lowers[start]);
  }

  free(csum);
  free(best);
  free(best_j);
  free(starts);
  return nb;
}