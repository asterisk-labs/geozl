from ._predictor import predictor

DeltaNInt, DeltaNDecoder = predictor(
    0x72D702, "geozl.lossless.delta_n",
    "geozl_delta_n_encode_auto", "geozl_delta_n_decode_auto", 4)
