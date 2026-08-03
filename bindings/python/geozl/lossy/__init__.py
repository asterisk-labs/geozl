from .quant_sqrt import Noise, fit_noise

# The three quantizers have no Python encoder classes yet, so there is nothing
# to register here. A frame carrying one still decodes through libgeozl, which
# registers its decoders from C.
_DECODERS = ()


def register_decoders(dctx):
    """Register the lossy decoders into an openzl.ext DCtx, for decoding in
    Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["Noise", "fit_noise", "register_decoders"]
