from importlib.metadata import PackageNotFoundError, version
from typing import Any

from . import lossless, lossy
from ._2d import Graph, ProfileResults, compress, decompress, graph, profile
from ._simd import simd_info

try:
    __version__ = version("geozl")
except PackageNotFoundError:
    __version__ = "0+unknown"


def register_decoders(dctx: Any) -> None:
    """Register all geozl decoders in an ``openzl.ext.DCtx``."""
    for decoder in lossless._DECODERS + lossy._DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["Graph", "ProfileResults", "lossless", "lossy", "compress",
           "decompress", "graph", "profile", "register_decoders",
           "simd_info", "__version__"]
