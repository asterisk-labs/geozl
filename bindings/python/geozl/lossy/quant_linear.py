from .._codec import quantizer

QuantLinear, QuantLinearDecoder = quantizer(
    0x72D781, "geozl.lossy.quant_linear", "quant_linear", ("step",),
    "the stream holds no finite non-zero sample to cut a grid against")

__all__ = ["QuantLinear", "QuantLinearDecoder"]
