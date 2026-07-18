import numpy as np

from ._ffi import _load_lib_full, _ptr, ffi
from .lossy.quant_linear import dtype_code

LOSSLESS = -1.0

# An OpenZL numeric stream is 1, 2, 4 or 8 bytes per element
_ELT_WIDTHS = frozenset((1, 2, 4, 8))


def compress(tile, *, method="full", width=None, max_error=None,
             format_version=None):
    """Compress a 2d tile into one OpenZL frame with geozl's native selector.

    tile            2d numpy array, or 1d with width given. The row width is
                    tile.shape[-1] unless width overrides it.
    method          "full" sweeps every predictor and keeps the smallest frame,
                    or a single predictor name: "delta_w", "delta_n", "planar",
                    "med", "average", "wp_static", "delta_1d".
    max_error       None or negative for lossless; a positive bound adds a lossy
                    quant_linear ahead of the predictors.
    format_version  OpenZL frame version, defaults to ext.MAX_FORMAT_VERSION.

    Returns the frame as bytes.
    """
    lib = _load_lib_full()
    arr = np.ascontiguousarray(tile)
    if width is None:
        if arr.ndim < 2:
            raise ValueError("give width for a 1d tile")
        width = arr.shape[-1]
    n = arr.size
    elt = arr.dtype.itemsize
    if elt not in _ELT_WIDTHS:
        raise ValueError(f"dtype {arr.dtype} is {elt} bytes per element, an "
                         f"OpenZL numeric stream is 1, 2, 4 or 8")

    if max_error is None or max_error < 0:
        err, code = LOSSLESS, 0
    else:
        code = dtype_code(arr.dtype)
        if code is None:
            raise ValueError(f"quant_linear does not support dtype {arr.dtype}")
        err = float(max_error)

    if format_version is None:
        from openzl import ext as _ext
        format_version = _ext.MAX_FORMAT_VERSION

    # The output buffer is allocated with a generous size, since the compressed
    # frame is usually smaller than the input tile. The size is based on the
    # input tile size, plus 50% for the frame header and any overhead.
    cap = 1024 + n * elt + n * elt // 2
    dst = np.empty(cap, np.uint8)
    out_size = ffi.new("size_t*")
    rc = lib.geozl_2d_compress_c(
        method.encode("utf-8"), int(width), err, int(code),
        int(format_version), _ptr(arr), n, elt,
        _ptr(dst), cap, out_size)
    if rc != 0:
        raise RuntimeError(f"geozl.compress failed for method {method!r} "
                           f"(ZL error code {rc})")
    return dst[:out_size[0]].tobytes()