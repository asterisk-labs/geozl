from ._predictor import predictor

PlanarInt, PlanarDecoder = predictor(
    0x72D703, "geozl.lossless.planar",
    "geozl_planar_encode_auto", "geozl_planar_decode_auto", 4)
