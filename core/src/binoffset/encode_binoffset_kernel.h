#ifndef GEOZL_CODECS_BINOFFSET_ENCODE_KERNEL_H
#define GEOZL_CODECS_BINOFFSET_ENCODE_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Power-of-two bins: bin = bit_length(v), offset = v with its top bit cleared.
// elt_width in {1,2,4,8}.
void binoffset_split(uint8_t *bins, void *offsets, const void *src,
                     size_t nb_elts, size_t elt_width);

// Explicit table, lowers ascending and padded to 256. nb_bins in 1..=255.
void binoffset_split_table(uint8_t *bins, void *offsets, const void *src,
                           size_t nb_elts, size_t elt_width,
                           const uint64_t lowers[256],
                           const uint8_t offset_bits[256], unsigned nb_bins);

// Fine histogram over sorted latents (pcodec histograms.rs). Up to 2^n_bins_log
// bins; out_lowers/out_uppers/out_counts must hold that many. Returns the count.
size_t binoffset_histogram(const uint64_t *sorted, size_t n, unsigned n_bins_log,
                           uint64_t *out_lowers, uint64_t *out_uppers,
                           uint32_t *out_counts);

// Optimal bin table from a fine histogram (pcodec bin_optimization.rs): the
// O(B^2) DP with its single-bin and trivial-offset shortcuts. Fine bins
// ascending, fine_counts[i] >= 1. meta_bits is the caller's per-bin cost. Writes
// up to max_bins (<= 255) bins; tail folds into the last. Returns the count.
unsigned binoffset_optimize_table(const uint64_t *fine_lowers,
                                  const uint64_t *fine_uppers,
                                  const uint32_t *fine_counts, size_t n_fine,
                                  double meta_bits, uint64_t *out_lowers,
                                  uint8_t *out_offset_bits, unsigned max_bins);


// split_table but offsets bit-packed to offset_bits[bin] bits. packed must be
// zeroed, holds nb_elts*elt_width bytes. Returns packed bytes used.
size_t binoffset_split_pack(uint8_t *bins, uint8_t *packed, const void *src,
                            size_t nb_elts, size_t elt_width,
                            const uint64_t lowers[256],
                            const uint8_t offset_bits[256], unsigned nb_bins);

#endif // GEOZL_CODECS_BINOFFSET_ENCODE_KERNEL_H