from .._codec import spatial_predictor

MedZigzag, MedZigzagDecoder = spatial_predictor(
    0x72D711, "geozl.lossless.med_zigzag",
    "med_zigzag_encode", "med_zigzag_decode")

__all__ = ["MedZigzag", "MedZigzagDecoder"]
