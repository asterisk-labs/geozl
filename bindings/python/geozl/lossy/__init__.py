from .quant_linear import QuantLinear, QuantLinearDecoder

_DECODERS = (QuantLinearDecoder,)


def register_decoders(dctx):
    """Register the lossy decoders into an openzl.ext DCtx, for decoding in
    Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["QuantLinear", "QuantLinearDecoder", "register_decoders"]
