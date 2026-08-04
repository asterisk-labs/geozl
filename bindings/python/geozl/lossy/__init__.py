from .quant_linear import QuantLinear, QuantLinearDecoder
from .quant_log import QuantLog, QuantLogDecoder
from .quant_sqrt import Noise, QuantSqrt, QuantSqrtDecoder, fit_noise

_DECODERS = (QuantLinearDecoder, QuantLogDecoder, QuantSqrtDecoder)


def register_decoders(dctx):
    """Register the lossy decoders into an openzl.ext DCtx, for decoding in
    Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = [
    "Noise", "QuantLinear", "QuantLinearDecoder", "QuantLog", "QuantLogDecoder",
    "QuantSqrt", "QuantSqrtDecoder", "fit_noise", "register_decoders",
]
