#include "lossy_recipe.h"

#include "quant_linear/encode_quant_linear_kernel.h"
#include "quant_linear/quant_linear_spec.h"
#include "quant_log/encode_quant_log_kernel.h"
#include "quant_log/quant_log_spec.h"
#include "quant_sqrt/encode_quant_sqrt_kernel.h"
#include "quant_sqrt/quant_sqrt_fit.h"
#include "quant_sqrt/quant_sqrt_spec.h"

#include "common/recipe_parse.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

// family lands last, so a recipe that did not parse stays NONE and reads as
// lossless rather than as a family sitting over an uninitialised spec.
int geozl_lossy_parse(const char *s, geozl_lossy_recipe *out, char *err,
                      size_t errSize) {
  if (out == NULL)
    return geozl_recipe_fail(err, errSize, "no recipe to write to");
  memset(out, 0, sizeof(*out));
  if (s == NULL || s[0] == '\0')
    return 0;

  if (strncmp(s, "LINEAR:", 7) == 0) {
    if (quant_linear_parse(s, &out->as.linear, err, errSize) != 0)
      return 1;
    out->family = GEOZL_LOSSY_LINEAR;
    return 0;
  }
  if (strncmp(s, "LOG:", 4) == 0) {
    if (quant_log_parse(s, &out->as.log, err, errSize) != 0)
      return 1;
    out->family = GEOZL_LOSSY_LOG;
    return 0;
  }
  if (strncmp(s, "SQRT:", 5) == 0) {
    if (quant_sqrt_parse(s, &out->as.sqrt, err, errSize) != 0)
      return 1;
    out->family = GEOZL_LOSSY_SQRT;
    return 0;
  }
  return geozl_recipe_fail(err, errSize,
              "error \"%s\": the families are LINEAR, LOG and SQRT, each "
              "followed by a colon",
              s);
}

int geozl_lossy_fit(geozl_lossy_recipe *r, const void *src, int dtype,
                    size_t width, size_t nbElts, char *err, size_t errSize) {
  if (r == NULL)
    return geozl_recipe_fail(err, errSize, "no recipe to fit");
  if (r->family != GEOZL_LOSSY_SQRT || r->as.sqrt.have_ab)
    return 0;
  if (width == 0 || nbElts % width != 0)
    return geozl_recipe_fail(err, errSize, "%zu samples do not divide into rows of %zu",
                nbElts, width);

  quant_sqrt_noise curve;
  if (quant_sqrt_fit(src, dtype, width, nbElts / width, &curve, err, errSize) !=
      0)
    return 1;
  r->as.sqrt.a = curve.a;
  r->as.sqrt.b = curve.b;
  r->as.sqrt.have_ab = 1;
  return 0;
}

int geozl_lossy_resolve(const geozl_lossy_recipe *r, const void *src, int dtype,
                        size_t nbElts, geozl_lossy_plan *out, char *err,
                        size_t errSize) {
  if (r == NULL || out == NULL)
    return geozl_recipe_fail(err, errSize, "no recipe to resolve");
  memset(out, 0, sizeof(*out));
  out->family = r->family;

  switch (r->family) {
  case GEOZL_LOSSY_NONE:
    return 0;

  case GEOZL_LOSSY_LINEAR: {
    quant_linear_stats sc;
    if (quant_linear_scan(src, dtype, nbElts, &sc) != 0)
      return geozl_recipe_fail(err, errSize,
                  "the raster holds no finite non-zero sample to cut a grid "
                  "against");
    return quant_linear_resolve(&r->as.linear, dtype, &sc, &out->as.linear, err,
                                errSize);
  }

  case GEOZL_LOSSY_LOG: {
    quant_log_stats sc;
    if (quant_log_scan(src, dtype, nbElts, &sc) != 0)
      return geozl_recipe_fail(err, errSize, "dtype %d is not a type quant_log knows", dtype);
    return quant_log_resolve(&r->as.log, dtype, &sc, &out->as.log, err,
                             errSize);
  }

  case GEOZL_LOSSY_SQRT: {
    quant_sqrt_stats sc;
    if (quant_sqrt_scan(src, dtype, nbElts, &sc) != 0)
      return geozl_recipe_fail(err, errSize, "the raster holds no finite sample");
    // NULL curve: geozl_lossy_fit has already written one into the recipe, or
    // the recipe came with A and B, or quant_sqrt refuses and says so.
    return quant_sqrt_resolve(&r->as.sqrt, dtype, &sc, NULL, &out->as.sqrt, err,
                              errSize);
  }
  }

  return geozl_recipe_fail(err, errSize, "recipe family %d is not one this build knows",
              (int)r->family);
}