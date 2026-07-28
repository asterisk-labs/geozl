from .quant import Quant, QuantDecoder

_DECODERS = (QuantDecoder,)


def register_decoders(dctx):
    """Register the lossy decoders into an openzl.ext DCtx, for decoding in
    Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["Quant", "QuantDecoder", "register_decoders"]
