from importlib.metadata import PackageNotFoundError, version

from . import lossless, lossy
from ._2d import compress, decompress, profile

try:
    __version__ = version("geozl")
except PackageNotFoundError:
    __version__ = "0+unknown"


def register_decoders(dctx):
    """Register every geozl decoder, lossless and lossy, into an openzl.ext DCtx.
    The counterpart of geozl_register_decoders on the C side."""
    lossless.register_decoders(dctx)
    lossy.register_decoders(dctx)


__all__ = ["lossless", "lossy", "compress", "decompress", "profile",
           "register_decoders", "__version__"]