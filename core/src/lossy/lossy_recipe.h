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

// Parse a LINEAR, LOG or SQRT recipe. NULL and "" select lossless mode.
int geozl_lossy_parse(const char *s, geozl_lossy_recipe *out, char *err,
                      size_t errSize);

// Fit missing SQRT A and B values from the raster. Other recipes are unchanged.
int geozl_lossy_fit(geozl_lossy_recipe *r, const void *src, int dtype,
                    size_t width, size_t nbElts, char *err, size_t errSize);

// Resolve a parsed recipe against the input samples.
int geozl_lossy_resolve(const geozl_lossy_recipe *r, const void *src, int dtype,
                        size_t nbElts, geozl_lossy_plan *out, char *err,
                        size_t errSize);

#endif // GEOZL_LOSSY_RECIPE_H
