from .._codec import spatial_predictor

Average, AverageDecoder = spatial_predictor(
    0x72D706, "geozl.lossless.average", "average_encode", "average_decode",
    """Predict each sample as the floored mean of its left and upper samples.

    ``width`` is the row length in samples. ``planes`` counts stacked images,
    and the predictor resets at the first row of each.
    """)

__all__ = ["Average", "AverageDecoder"]
