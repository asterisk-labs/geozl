from importlib.metadata import PackageNotFoundError, version

from . import lossless, lossy, train

try:
    __version__ = version("geozl")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["lossless", "lossy", "train", "__version__"]