from .._codec import spatial_predictor

Planar, PlanarDecoder = spatial_predictor(
    0x72D703, "geozl.lossless.planar", "planar_encode", "planar_decode",
    """Predict each sample as left plus above minus upper left.

    ``width`` is the row length in samples. ``planes`` counts stacked images,
    and the predictor resets at the first row of each.
    """)

__all__ = ["Planar", "PlanarDecoder"]
