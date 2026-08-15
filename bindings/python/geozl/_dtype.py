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


def nodata_bits(value: Any, dtype: DTypeLike) -> int:
    """Return a nodata value's bit pattern at ``dtype``."""
    dt = np.dtype(dtype)
    if dtype_code(dt) is None:
        raise ValueError(f"geozl has no code for dtype {dt}")
    if dt == np.dtype("float16"):
        raise ValueError("a half float carries no sentinel, NaN still works "
                         "on one")
    v = np.asarray(value)
    if dt.kind in "iu" and v.dtype.kind == "f" and v != np.floor(v):
        raise ValueError(f"nodata {value} is not a whole number, it cannot be "
                         f"a sentinel on {dt}")
    with np.errstate(over="raise"):
        # out of range has to raise, not wrap or go infinite
        packed = np.asarray(value, dtype=dt)
    return int(packed.view(np.dtype(f"u{dt.itemsize}")))
