from typing import Any

from ._noise import Noise, fit_noise

__all__ = ["Noise", "fit_noise"]

# The quantizer nodes subclass openzl.ext types. Without openzl only the noise
# fit is here, which a tiled SQRT graph still needs for its A and B.
try:
    from .quant_linear import QuantLinear, QuantLinearDecoder
    from .quant_log import QuantLog, QuantLogDecoder
    from .quant_sqrt import QuantSqrt, QuantSqrtDecoder
except ModuleNotFoundError as exc:
    if exc.name != "openzl":
        raise
    _DECODERS: tuple[type, ...] = ()
    _NODES = ("QuantLinear", "QuantLinearDecoder", "QuantLog",
              "QuantLogDecoder", "QuantSqrt", "QuantSqrtDecoder")

    def __getattr__(name: str) -> Any:
        # Name the missing package, as importing the node's module would.
        if name in _NODES:
            raise ModuleNotFoundError(
                f"geozl.lossy.{name} needs the openzl package", name="openzl")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
else:
    _DECODERS = (QuantLinearDecoder, QuantLogDecoder, QuantSqrtDecoder)
    __all__ += ["QuantLinear", "QuantLinearDecoder", "QuantLog",
                "QuantLogDecoder", "QuantSqrt", "QuantSqrtDecoder"]
