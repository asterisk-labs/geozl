import math
import numbers
from decimal import Decimal
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

_CODES: dict[np.dtype, int] = {
    np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2,
    np.dtype("uint64"): 3, np.dtype("int8"): 4, np.dtype("int16"): 5,
    np.dtype("int32"): 6, np.dtype("int64"): 7, np.dtype("float16"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10,
}

# Width per code, off the table above so the two cannot drift.
_WIDTH = tuple(dt.itemsize
               for dt, _c in sorted(_CODES.items(), key=lambda kv: kv[1]))


def dtype_code(dtype: DTypeLike) -> int | None:
    """Return the wire code for a native NumPy dtype, if supported."""
    return _CODES.get(np.dtype(dtype))


def dtype_width(code: int) -> int | None:
    """Return the element width for a dtype code, if valid."""
    return _WIDTH[code] if 0 <= code < len(_WIDTH) else None


def refuse_object(dtype: DTypeLike) -> None:
    """Raise for a dtype whose elements are references, not samples."""
    if np.dtype(dtype).hasobject:
        raise ValueError("object arrays hold references, not samples")


def _mark(dt: np.dtype, at: int, covered: bytearray) -> None:
    if dt.fields is not None:
        for field in dt.fields.values():
            _mark(field[0], at + field[1], covered)
    elif dt.subdtype is not None:
        sub, shape = dt.subdtype
        for i in range(math.prod(shape)):
            _mark(sub, at + i * sub.itemsize, covered)
    else:
        covered[at:at + dt.itemsize] = b"\1" * dt.itemsize


def refuse_padding(dtype: DTypeLike) -> None:
    """Raise for a structured dtype with bytes no field covers, which a frame
    would store undefined."""
    dt = np.dtype(dtype)
    if dt.fields is None:
        return
    covered = bytearray(dt.itemsize)
    _mark(dt, 0, covered)
    if not all(covered):
        raise ValueError("a structured dtype must have no padding bytes")


def refuse_non_number(value: Any) -> None:
    """Raise unless ``value`` is one real number, NaN included."""
    if isinstance(value, np.ndarray):
        number = value.ndim == 0 and value.dtype.kind in "iuf"
    else:
        number = (isinstance(value, (numbers.Real, Decimal)) and
                  not isinstance(value, (bool, np.bool_)))
    if not number:
        raise TypeError("nodata must be a number")


def is_nan(value: Any) -> bool:
    """Return whether a number is NaN, a signaling Decimal one included."""
    if isinstance(value, Decimal):
        return value.is_nan()
    return bool(value != value)


def nodata_bits(value: Any, dtype: DTypeLike) -> int:
    """Return a nodata value's bit pattern at ``dtype``."""
    refuse_non_number(value)
    # Quiet signaling Decimal NaNs before comparison.
    if isinstance(value, Decimal) and value.is_nan():
        value = float("-nan" if value.is_signed() else "nan")
    dt = np.dtype(dtype)
    if dtype_code(dt) is None:
        raise ValueError(f"geozl has no code for dtype {dt}")
    if dt == np.dtype("float16"):
        raise ValueError("a half float carries no sentinel, NaN still works "
                         "on one")
    # NumPy casts its own values unchecked. At the same dtype there is no cast,
    # and a Python float would quiet a signaling NaN.
    if isinstance(value, (np.generic, np.ndarray)) and value.dtype != dt:
        value = value.item()
    if dt.kind in "iu":
        # Compare before casting so Fraction and Decimal values are not truncated.
        x = value.item() if isinstance(value, (np.generic, np.ndarray)) else value
        if x != x:
            raise ValueError(f"a {dt} raster cannot use NaN as nodata")
        if abs(x) != math.inf and x != int(x):
            raise ValueError(f"nodata {value} is not a whole number, it cannot "
                             f"be a sentinel on {dt}")
        if not np.iinfo(dt).min <= x <= np.iinfo(dt).max:
            raise ValueError("nodata does not fit the dtype")
        packed = np.asarray(int(x), dtype=dt)
    else:
        try:
            with np.errstate(over="raise"):
                packed = np.asarray(value, dtype=dt)
        except (OverflowError, FloatingPointError):
            raise ValueError("nodata does not fit the dtype") from None
        # NumPy can turn an oversized Decimal into infinity without warning.
        if np.isinf(packed) and abs(np.asarray(value).item()) != math.inf:
            raise ValueError("nodata does not fit the dtype")
    return int(packed.view(np.dtype(f"u{dt.itemsize}")))
