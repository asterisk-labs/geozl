from .._codec import spatial_predictor

MedZigzag, MedZigzagDecoder = spatial_predictor(
    0x72D711, "geozl.lossless.med_zigzag",
    "med_zigzag_encode", "med_zigzag_decode",
    """Run ``Med`` and zigzag the residuals in one pass.

    The output equals ``Med`` followed by OpenZL's zigzag node. ``width`` and
    ``planes`` are as in ``Med``.
    """)

__all__ = ["MedZigzag", "MedZigzagDecoder"]
