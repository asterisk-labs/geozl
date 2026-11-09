from .._codec import spatial_predictor

PlanarZigzag, PlanarZigzagDecoder = spatial_predictor(
    0x72D70F, "geozl.lossless.planar_zigzag",
    "planar_zigzag_encode", "planar_zigzag_decode",
    """Run ``Planar`` and zigzag the residuals in one pass.

    The output equals ``Planar`` followed by OpenZL's zigzag node. ``width``
    and ``planes`` are as in ``Planar``.
    """)

__all__ = ["PlanarZigzag", "PlanarZigzagDecoder"]
