from .._codec import spatial_predictor

DeltaW, DeltaWDecoder = spatial_predictor(
    0x72D701, "geozl.lossless.delta_w", "delta_w_encode", "delta_w_decode")

__all__ = ["DeltaW", "DeltaWDecoder"]