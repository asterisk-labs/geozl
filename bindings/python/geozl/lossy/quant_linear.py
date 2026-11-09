from .._codec import quantizer

QuantLinear, QuantLinearDecoder = quantizer(
    0x72D781, "geozl.lossy.quant_linear", "quant_linear", ("step",),
    "the stream holds no finite non-zero sample to cut a grid against",
    """Quantize to a uniform grid with an absolute error bound.

    ``recipe`` is a ``LINEAR:MAX_ERROR=V`` string and ``dtype`` the sample
    type. The grid is resolved from each stream the node encodes.
    """)

__all__ = ["QuantLinear", "QuantLinearDecoder"]
