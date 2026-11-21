from typing import Any

from .._codec import quantizer
from .._ffi import ffi
from ._noise import Noise, fit_noise  # their import path before _noise


def _needs_curve(spec: Any) -> str | None:
    if spec.have_ab:
        return None
    return "has no A and B, which a node cannot fit, so use Noise.recipe"


# NULL curve. resolve reads A and B off the recipe first, so one passed here
# alongside them would be dropped. Noise.recipe is the way in.
QuantSqrt, QuantSqrtDecoder = quantizer(
    0x72D783, "geozl.lossy.quant_sqrt", "quant_sqrt", ("step", "offset"),
    "the stream holds no finite sample",
    """Quantize on a square root grid with an error bound in noise sigmas.

    ``recipe`` is a ``SQRT:MAX_ERROR=kN,A=a,B=b`` string, as
    ``Noise.recipe`` writes it, and ``dtype`` the sample type. A recipe
    without ``A`` and ``B`` raises ``ValueError``.
    """,
    extra=(ffi.NULL,), check=_needs_curve)

__all__ = ["Noise", "QuantSqrt", "QuantSqrtDecoder", "fit_noise"]
