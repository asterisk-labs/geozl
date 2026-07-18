from .delta_w import DeltaW, DeltaWDecoder
from .delta_n import DeltaN, DeltaNDecoder
from .planar import Planar, PlanarDecoder
from .med import Med, MedDecoder
from .average import Average, AverageDecoder
from .wp_static import WpStatic, WpStaticDecoder
from .deinterleave import Deinterleave, DeinterleaveDecoder, component_dtype

_DECODERS = (DeltaWDecoder, DeltaNDecoder, PlanarDecoder, MedDecoder,
             AverageDecoder, WpStaticDecoder, DeinterleaveDecoder)


def register_decoders(dctx):
    """Register the lossless decoders into an openzl.ext DCtx, for decoding in
    Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = [
    "Average", "AverageDecoder", "DeltaN", "DeltaNDecoder", "DeltaW",
    "DeltaWDecoder", "Deinterleave", "DeinterleaveDecoder", "Med", "MedDecoder",
    "Planar", "PlanarDecoder", "WpStatic", "WpStaticDecoder", "component_dtype",
    "register_decoders",
]