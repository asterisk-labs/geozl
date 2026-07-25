#ifndef GEOZL_CODECS_NODATA_ENCODE_BINDING_H
#define GEOZL_CODECS_NODATA_ENCODE_BINDING_H

#include "common/graph_num1to2.h"
#include "nodata/nodata_wire.h"
#include "openzl/zl_ctransform.h"

// Encoder side selectors. The wire header never carries these, by the time it
// is written the pattern is known and both modes look the same to a reader.
#define GEOZL_NODATA_PARAM_WIDTH 1
#define GEOZL_NODATA_PARAM_MODE 2
#define GEOZL_NODATA_PARAM_VALUE 3

#define GEOZL_NODATA_MODE_VALUE 1 // pattern comes in GEOZL_NODATA_PARAM_VALUE
#define GEOZL_NODATA_MODE_NAN 2   // encoder finds the non-finite pattern itself

ZL_Report EI_geozl_nodata(ZL_Encoder *eictx, const ZL_Input *in);

#define EI_NODATA(id)                                                          \
  {                                                                            \
    .gd = GEOZL_NUM1TO2_GRAPH(id), .transform_f = EI_geozl_nodata,             \
    .name = "geozl.lossless.nodata",                                           \
  }

#endif // GEOZL_CODECS_NODATA_ENCODE_BINDING_H
