from .._codec import spatial_predictor

PlanarZigzag, PlanarZigzagDecoder = spatial_predictor(
    0x72D70F, "geozl.lossless.planar_zigzag",
    "planar_zigzag_encode", "planar_zigzag_decode")

__all__ = ["PlanarZigzag", "PlanarZigzagDecoder"]
