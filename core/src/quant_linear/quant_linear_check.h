// What a parameter block has to satisfy, read by both ends. Written twice
// before, and the two copies had drifted.

#ifndef GEOZL_CODECS_QUANT_LINEAR_CHECK_H
#define GEOZL_CODECS_QUANT_LINEAR_CHECK_H

#include "quant_linear_dtype.h"

#include "geozl/quant_linear_params.h"

#include <float.h>
#include <math.h>

// A normal step, not merely finite and positive. A subnormal one passes both
// ends and then saturates every index at the type maximum, in silence.
static inline int quant_linear_params_ok(const quant_linear_params *p,
                                         int dtype) {
  if (p == NULL || !QL_DTYPE_OK(dtype))
    return 0;
  if ((p->flags & ~(unsigned)QUANT_LINEAR_FLAGS_KNOWN) != 0)
    return 0;
  if (!isfinite(p->step) || !(p->step >= DBL_MIN))
    return 0;

  const int values = (p->flags & QUANT_LINEAR_FLAG_STORE_VALUES) != 0;
  if (dtype <= QL_LAST_INT)
    return values;
  // A float frame carries a reconstruction only at a whole step.
  return !values || floor(p->step) == p->step;
}

#endif // GEOZL_CODECS_QUANT_LINEAR_CHECK_H
