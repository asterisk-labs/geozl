from .._codec import spatial_predictor

DeltaN, DeltaNDecoder = spatial_predictor(
    0x72D702, "geozl.lossless.delta_n", "delta_n_encode", "delta_n_decode",
    """Predict each sample from the sample above it.

    ``width`` is the row length in samples. ``planes`` counts stacked images,
    and the first row of each is stored as it is.
    """)

__all__ = ["DeltaN", "DeltaNDecoder"]
