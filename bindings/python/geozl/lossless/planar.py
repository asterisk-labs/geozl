from .._codec import spatial_predictor

Planar, PlanarDecoder = spatial_predictor(
    0x72D703, "geozl.lossless.planar", "planar_encode", "planar_decode")

__all__ = ["Planar", "PlanarDecoder"]