from .._codec import spatial_predictor

DeltaN, DeltaNDecoder = spatial_predictor(
    0x72D702, "geozl.lossless.delta_n", "delta_n_encode", "delta_n_decode")

__all__ = ["DeltaN", "DeltaNDecoder"]