
from .._codec import spatial_predictor
 
Average, AverageDecoder = spatial_predictor(
    0x72D706, "geozl.lossless.average", "average_encode", "average_decode")
 
__all__ = ["Average", "AverageDecoder"]