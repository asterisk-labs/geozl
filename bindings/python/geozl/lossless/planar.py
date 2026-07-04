from ._predictor import predictor

PlanarInt, PlanarDecoder = predictor(
    0x72D703, "geozl.lossless.planar",
    "geozl_planar_encode", "geozl_planar_decode")
