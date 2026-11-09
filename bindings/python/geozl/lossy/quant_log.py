from .._codec import quantizer

QuantLog, QuantLogDecoder = quantizer(
    0x72D782, "geozl.lossy.quant_log", "quant_log", ("step",),
    "dtype {dtype} is not a type quant_log knows",
    """Quantize on a logarithmic grid with a relative error bound.

    ``recipe`` is a ``LOG:MAX_ERROR=P%`` string and ``dtype`` the sample type.
    The grid is resolved from each stream the node encodes.
    """)

__all__ = ["QuantLog", "QuantLogDecoder"]
