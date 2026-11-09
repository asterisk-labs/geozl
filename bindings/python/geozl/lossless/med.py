from .._codec import spatial_predictor

Med, MedDecoder = spatial_predictor(
    0x72D705, "geozl.lossless.med", "med_encode", "med_decode",
    """Predict each sample with the median edge detector of LOCO-I.

    ``width`` is the row length in samples. ``planes`` counts stacked images,
    and the predictor resets at the first row of each.
    """)

__all__ = ["Med", "MedDecoder"]
