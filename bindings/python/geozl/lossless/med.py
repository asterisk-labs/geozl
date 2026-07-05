from ._predictor import predictor

MedInt, MedDecoder = predictor(
    0x72D705, "geozl.lossless.med",
    "geozl_med_encode_auto", "geozl_med_decode_auto", 4)
