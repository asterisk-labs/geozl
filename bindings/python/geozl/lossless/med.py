from .._codec import spatial_predictor

Med, MedDecoder = spatial_predictor(
    0x72D705, "geozl.lossless.med", "med_encode", "med_decode")

__all__ = ["Med", "MedDecoder"]