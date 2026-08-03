#ifndef GEOZL_LOSSY_RECIPE_H
#define GEOZL_LOSSY_RECIPE_H

#include "geozl/quant_linear_params.h"
#include "geozl/quant_log_params.h"
#include "geozl/quant_sqrt_params.h"

#include <stddef.h>

typedef enum {
  GEOZL_LOSSY_NONE = 0,
  GEOZL_LOSSY_LINEAR,
  GEOZL_LOSSY_LOG,
  GEOZL_LOSSY_SQRT
} geozl_lossy_family;

typedef struct {
  geozl_lossy_family family;
  union {
    quant_linear_spec linear;
    quant_log_spec log;
    quant_sqrt_spec sqrt;
  } as;
} geozl_lossy_recipe;

typedef struct {
  geozl_lossy_family family;
  union {
    quant_linear_params linear;
    quant_log_params log;
    quant_sqrt_params sqrt;
  } as;
} geozl_lossy_plan;

// A recipe into a frame's parameters, in three passes that read different
// things. parse reads the string, fit reads the raster as an image, resolve
// reads the samples. Nothing here touches OpenZL; the plan becomes a node
// through geozl_node_lossy in lossy_node.h.

// "LINEAR:MAX_ERROR=V", "LOG:MAX_ERROR=P%", "SQRT:MAX_ERROR=VN". NULL or "" is
// lossless. The prefix picks the family, the rest is that codec's own grammar
// and its spec.md documents it.
int geozl_lossy_parse(const char *s, geozl_lossy_recipe *out, char *err,
                      size_t errSize);

// Fills in what the recipe left to the data, which today is a SQRT recipe
// carrying no A and B. Separate from resolve because it needs rows and columns
// rather than a run of samples. A no-op for everything else.
//
// The curve it measures belongs to this raster, so tiles fitted one at a time do
// not share a grid. Measure once over the product, write A and B into the
// recipe, skip this. quant_sqrt/spec.md has the rest.
int geozl_lossy_fit(geozl_lossy_recipe *r, const void *src, int dtype,
                    size_t width, size_t nbElts, char *err, size_t errSize);

// Scans the samples and cuts the grid.
int geozl_lossy_resolve(const geozl_lossy_recipe *r, const void *src, int dtype,
                        size_t nbElts, geozl_lossy_plan *out, char *err,
                        size_t errSize);

#endif // GEOZL_LOSSY_RECIPE_H