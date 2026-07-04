from .delta_n import DeltaNDecoder, DeltaNInt
from .delta_w import DeltaWDecoder, DeltaWInt
from .planar import PlanarDecoder, PlanarInt

_DECODERS = (DeltaWDecoder, DeltaNDecoder, PlanarDecoder)


def register_decoders(dctx) -> None:
    """Register the lossless decoders into an openzl.ext DCtx, for decoding
    frames in pure Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = [
    "DeltaNDecoder", "DeltaNInt", "DeltaWDecoder", "DeltaWInt",
    "PlanarDecoder", "PlanarInt", "register_decoders",
]
