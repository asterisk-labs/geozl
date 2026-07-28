#ifndef GEOZL_CODECS_QUANT_SPEC_H
#define GEOZL_CODECS_QUANT_SPEC_H

#include "quant_curve.h" // quant_params, quant_spec

#include <stddef.h>

// The recipes quant_spec_parse accepts, all of them.
//
//   NULL or ""            lossless
//   "abs:V"               |x - x^| <= V
//   "rel:P%"              |x - x^| <= (P/100) * |x|
//   "shot:a=A,b=B,k=K"    |x - x^| <= K * sqrt(A + B*x)
//
// The percent sign in rel is required. "rel:1" would otherwise read as a bound
// of one, a hundred percent, a likely typo for one percent that would not
// fail.

// Parse. Returns 0, or nonzero with the reason in err.
int quant_spec_parse(const char *s, quant_spec *out, char *err, size_t errSize);

// Finish the parameters against the tile. The log curve anchors its grid on the
// smallest magnitude present, so they are only complete once the data has been
// scanned. Returns 0, or nonzero with the reason in err.
int quant_spec_resolve(const quant_spec *sp, int dtype, double minAbs,
                       double maxAbs, int anyNegative, quant_params *out,
                       char *err, size_t errSize);

// Largest index magnitude the stream can carry at this dtype, which is the
// sample width, signed for a float original.
double quant_index_max(int dtype);

// Worst error in the round trip, as a fraction of the bound the recipe
// declared. @dec is what quant_decode gave back for the stream quant_encode
// produced, so this measures the real decoder rather than the algebra behind
// it, which is the point: a cause nobody predicted still shows up as a number
// above one. Returns 0 when the frame holds the bound it declares.
int quant_verify(const void *src, const void *dec, const quant_spec *sp,
                 int dtype, size_t nbElts, double *worst);

#endif // GEOZL_CODECS_QUANT_SPEC_H