from importlib.metadata import PackageNotFoundError, version
from typing import Any

from . import lossy
from ._2d import Graph, ProfileResults, compress, decompress, graph, profile
from ._coeffs import coeffs
from ._error import GeoZLError
from ._simd import simd_info

# OpenZL publishes wheels for a few platforms only. Where it has none the
# high-level API still runs on the OpenZL inside libgeozl, and the codec classes,
# which subclass openzl.ext types, are the part that is missing. geozl.lossy
# then holds only the noise fit, which runs on the kernels.
try:
    from . import lossless
except ModuleNotFoundError as exc:
    if exc.name != "openzl":
        raise
    _LOW_LEVEL: list[str] = []
else:
    _LOW_LEVEL = ["lossless"]

try:
    __version__ = version("geozl")
except PackageNotFoundError:
    __version__ = "0+unknown"


def register_decoders(dctx: Any) -> None:
    """Register all geozl decoders in an ``openzl.ext.DCtx``."""
    if not _LOW_LEVEL:
        raise ModuleNotFoundError(
            "geozl.register_decoders needs the openzl package", name="openzl")
    for decoder in lossless._DECODERS + lossy._DECODERS:
        dctx.register_custom_decoder(decoder())


__all__ = ["GeoZLError", "Graph", "ProfileResults", *_LOW_LEVEL, "lossy",
           "coeffs", "compress", "decompress", "graph", "profile",
           "register_decoders", "simd_info", "__version__"]
