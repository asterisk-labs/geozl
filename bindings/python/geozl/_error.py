# ZL_ErrorCode_parameter_invalid and ZL_ErrorCode_graph_invalid.
_INVALID = frozenset((21, 30))


class GeoZLError(RuntimeError):
    """A library refusal, with its ZL error code when available."""

    code: int | None

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class _InvalidError(GeoZLError, ValueError):
    """An invalid graph or parameter, also catchable as ValueError."""


def refusal(message: str, code: int) -> GeoZLError:
    """Classify a refused request. Decoding errors use GeoZLError directly."""
    cls = _InvalidError if code in _INVALID else GeoZLError
    return cls(message, code)
