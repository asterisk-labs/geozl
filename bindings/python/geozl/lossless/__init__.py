from .._codec import spatial_predictor
from .deinterleave import Deinterleave, DeinterleaveDecoder, component_dtype
from .wp_static import WpStatic, WpStaticDecoder

# Each spatial predictor is one numeric in, one out, with a row width, and they
# differ only in the kernel. Fields are CTid, name, encode kernel, decode kernel.
# The CTids match core/include/geozl/ctids.h.
DeltaW, DeltaWDecoder = spatial_predictor(
    0x72D701, "geozl.lossless.delta_w", "delta_w_encode", "delta_w_decode")
DeltaN, DeltaNDecoder = spatial_predictor(
    0x72D702, "geozl.lossless.delta_n", "delta_n_encode", "delta_n_decode")
Planar, PlanarDecoder = spatial_predictor(
    0x72D703, "geozl.lossless.planar", "planar_encode", "planar_decode")
Med, MedDecoder = spatial_predictor(
    0x72D705, "geozl.lossless.med", "med_encode", "med_decode")
Average, AverageDecoder = spatial_predictor(
    0x72D706, "geozl.lossless.average", "average_encode", "average_decode")

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
