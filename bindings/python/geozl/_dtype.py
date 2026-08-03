"""The element type codes geozl carries in a codec header.

Mirrors geozl_dtype in core/include/geozl/dtype.h, and the numbers are frozen
because a frame written today is read against them.
"""

import numpy as np

_CODES = {
    np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2,
    np.dtype("uint64"): 3, np.dtype("int8"): 4, np.dtype("int16"): 5,
    np.dtype("int32"): 6, np.dtype("int64"): 7, np.dtype("float16"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10,
}

# Element width per code, derived from the table above so the two cannot drift.
# Codes are 0..N-1, so a tuple indexes straight by code.
_WIDTH = tuple(dt.itemsize
               for dt, _c in sorted(_CODES.items(), key=lambda kv: kv[1]))


def dtype_code(dtype):
    """The wire code for dtype, or None when geozl has no kernel for it. Keyed
    on the numpy dtype, so a byte-swapped array is refused rather than read as
    native."""
    return _CODES.get(np.dtype(dtype))


def dtype_width(code):
    """Bytes per element for a wire code."""
    return _WIDTH[code]
