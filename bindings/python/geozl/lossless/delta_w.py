from ._predictor import predictor

DeltaWInt, DeltaWDecoder = predictor(
    0x72D701, "geozl.lossless.delta_w",
    "geozl_delta_w_encode", "geozl_delta_w_decode")
