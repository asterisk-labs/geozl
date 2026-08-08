from importlib.metadata import PackageNotFoundError, version
from typing import Any

from . import lossless, lossy
from ._2d import Graph, compress, decompress, graph, profile
from ._simd import simd_info

try:
    __version__ = version("geozl")
except PackageNotFoundError:
    __version__ = "0+unknown"


def register_decoders(dctx: Any) -> None:
    """Register every geozl decoder, lossless and lossy, into an openzl.ext
    DCtx. The counterpart of geozl_register_decoders on the C side."""
    for decoder in lossless._DECODERS + lossy._DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["Graph", "lossless", "lossy", "compress", "decompress",
           "graph", "profile", "register_decoders", "simd_info",
           "__version__"]