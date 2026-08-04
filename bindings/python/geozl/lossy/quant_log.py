from .._codec import quantizer

QuantLog, QuantLogDecoder = quantizer(
    0x72D782, "geozl.lossy.quant_log", "quant_log", ("step",),
    "dtype {dtype} is not a type quant_log knows")

__all__ = ["QuantLog", "QuantLogDecoder"]
