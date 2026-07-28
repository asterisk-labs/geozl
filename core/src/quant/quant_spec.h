#ifndef GEOZL_CODECS_QUANT_SPEC_H
#define GEOZL_CODECS_QUANT_SPEC_H

#include "quant_curve.h" // quant_params

#include <stddef.h>

// The error argument of the 2d API, parsed. A string rather than a struct for
// the same reason method is one: it crosses every entry point unchanged, so
// compress, bench and profile cannot end up describing different errors.
//
//   NULL or ""            lossless
//   "abs:V"               |x - x^| <= V
//   "rel:P%"              |x - x^| <= (P/100) * |x|
//   "shot:a=A,b=B,k=K"    |x - x^| <= K * sqrt(A + B*x)
//
// The percent sign in rel is required. Without it "rel:1" reads as a bound of
// one, a hundred percent, which is a plausible typo for one percent and would
// quantize a hundred times coarser than intended without failing.
typedef enum {
  QUANT_SPEC_LOSSLESS = 0,
  QUANT_SPEC_EXPLICIT = 1
} quant_spec_mode;

typedef struct {
  unsigned char mode;
  unsigned char curve;
  double abs_err;
  double rel_err;
  double shot_a, shot_b, shot_k;
} quant_spec;

// Parse. Returns 0, or nonzero with the reason in err.
int quant_spec_parse(const char *s, quant_spec *out, char *err, size_t errSize);

// Finish the parameters against the tile. The log curve anchors its grid on the
// smallest magnitude present, so the numbers are only complete once the data
// has been scanned. Returns 0, or nonzero with the reason in err.
int quant_spec_resolve(const quant_spec *sp, int dtype, double minAbs,
                       double maxAbs, int anyNegative, quant_params *out,
                       char *err, size_t errSize);

// Largest index magnitude the stream can carry at this dtype, which is the
// sample width, signed for a float original.
double quant_index_max(int dtype);

#endif // GEOZL_CODECS_QUANT_SPEC_H
