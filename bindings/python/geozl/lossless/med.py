from ._predictor import predictor

MedInt, MedDecoder = predictor(
    0x72D705, "geozl.lossless.med",
    "geozl_med_encode", "geozl_med_decode")