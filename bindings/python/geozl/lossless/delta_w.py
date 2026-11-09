from .._codec import spatial_predictor

DeltaW, DeltaWDecoder = spatial_predictor(
    0x72D701, "geozl.lossless.delta_w", "delta_w_encode", "delta_w_decode",
    """Predict each sample from its left neighbour.

    ``width`` is the row length in samples, and the first sample of each row
    is stored as it is. The predictor never reads the row above, so it takes
    no plane count.
    """,
    supports_planes=False)

__all__ = ["DeltaW", "DeltaWDecoder"]
