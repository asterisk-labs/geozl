import numpy as np

from ._ffi import ffi, _load_lib_full

# quant_linear dtype codes, mirrors ql_dtype in graph_quant_linear.h.
_QL_DTYPE = {
    np.dtype("uint8"): 0, np.dtype("uint16"): 1, np.dtype("uint32"): 2,
    np.dtype("uint64"): 3, np.dtype("int8"): 4, np.dtype("int16"): 5,
    np.dtype("int32"): 6, np.dtype("int64"): 7, np.dtype("float16"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10,
}

LOSSLESS = -1.0


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

    if max_error is None or max_error < 0:
        err, dtype_code = LOSSLESS, 0
    else:
        dtype_code = _QL_DTYPE.get(arr.dtype)
        if dtype_code is None:
            raise ValueError(f"quant_linear does not support dtype {arr.dtype}")
        err = float(max_error)

    if format_version is None:
        from openzl import ext as _ext
        format_version = _ext.MAX_FORMAT_VERSION

    cap = 1024 + n * elt + n * elt // 2  # generous, a frame never exceeds this
    dst = bytearray(cap)
    out_size = ffi.new("size_t*")
    rc = lib.geozl_2d_compress_c(
        method.encode("utf-8"), int(width), err, int(dtype_code),
        int(format_version), ffi.from_buffer(arr), n, elt,
        ffi.from_buffer(dst), cap, out_size)
    if rc != 0:
        raise RuntimeError(f"geozl.compress failed for method {method!r} "
                           f"(ZL error code {rc})")
    return bytes(dst[:out_size[0]])