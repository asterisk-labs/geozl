from ._predictor import predictor

AverageInt, AverageDecoder = predictor(
    0x72D706, "geozl.lossless.average",
    "geozl_average_encode", "geozl_average_decode")