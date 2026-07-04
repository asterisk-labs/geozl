#ifndef GEOZL_CODECS_MED_ENCODE_BINDING_H
#define GEOZL_CODECS_MED_ENCODE_BINDING_H

#include "graph_med.h"
#include "openzl/zl_ctransform.h"

ZL_Report EI_geozl_med(ZL_Encoder* eictx, const ZL_Input* in);

static const ZL_TypedEncoderDesc med_encoder_desc = {
    .gd          = MED_GRAPH,
    .transform_f = EI_geozl_med,
    .name        = "geozl.lossless.med",
};

#endif // GEOZL_CODECS_MED_ENCODE_BINDING_H
