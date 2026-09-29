from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ._2d import Graph, ProfileResults, compress, decompress, graph, profile
from ._coeffs import coeffs
from ._simd import simd_info

# OpenZL publishes wheels for a few platforms only. Where it has none the
# high-level API still runs on the OpenZL inside libgeozl, and the codec classes,
# which subclass openzl.ext types, are the part that is missing.
try:
    from . import lossless, lossy
except ModuleNotFoundError as exc:
    if exc.name != "openzl":
        raise
    _LOW_LEVEL: list[str] = []
else:
    _LOW_LEVEL = ["lossless", "lossy"]

try:
    __version__ = version("geozl")
except PackageNotFoundError:
    __version__ = "0+unknown"


def register_decoders(dctx: Any) -> None:
    """Register all geozl decoders in an ``openzl.ext.DCtx``."""
    for decoder in lossless._DECODERS + lossy._DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["Graph", "ProfileResults", *_LOW_LEVEL, "coeffs",
           "compress", "decompress", "graph", "profile", "register_decoders",
           "simd_info", "__version__"]
