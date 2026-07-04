from .complex import (
    ComplexSplitDecoder,
    complex_split,
    component_dtype,
    decompress_complex,
)
from .delta_n import DeltaNDecoder, DeltaNInt
from .delta_w import DeltaWDecoder, DeltaWInt
from .average import AverageDecoder, AverageInt
from .med import MedDecoder, MedInt
from .planar import PlanarDecoder, PlanarInt

_DECODERS = (DeltaWDecoder, DeltaNDecoder, PlanarDecoder, MedDecoder,
             AverageDecoder, ComplexSplitDecoder)


def register_decoders(dctx) -> None:
    """Register the lossless decoders into an openzl.ext DCtx, for decoding
    frames in pure Python. A C reader registers them through libgeozl instead."""
    for decoder in _DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = [
    "AverageDecoder", "AverageInt", "ComplexSplitDecoder", "DeltaNDecoder",
    "DeltaNInt", "DeltaWDecoder", "DeltaWInt", "MedDecoder", "MedInt",
    "PlanarDecoder", "PlanarInt", "complex_split", "component_dtype",
    "decompress_complex", "register_decoders",
]