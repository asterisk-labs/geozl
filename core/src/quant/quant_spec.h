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

// Encode, decode what was just written and measure it, tightening the step
// until the round trip holds the bound the recipe declared. That measurement
// is what an encoder writes a frame on, so every encoder comes through here
// instead of carrying its own copy. @idx and @chk are caller-owned scratch of
// nbElts
// elements, and @idx holds the index stream when this returns 0. @p is the
// resolved parameters going in and the ones the frame was written with coming
// out, so the header takes its step from here and not from before the call.
// Returns 0 when the frame holds, 1 when no step this side of zero makes it,
// and -1 when the kernels refused parameters the resolver produced.
int quant_fit(void *idx, void *chk, const void *src, const quant_spec *sp,
              quant_params *p, int dtype, size_t nbElts);

#endif // GEOZL_CODECS_QUANT_SPEC_H