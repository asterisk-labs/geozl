from .quant_linear import QuantLinearDecoder, quant_linear

_DECODERS = (QuantLinearDecoder,)


def register_decoders(dctx) -> None:
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["QuantLinearDecoder", "quant_linear", "register_decoders"]
