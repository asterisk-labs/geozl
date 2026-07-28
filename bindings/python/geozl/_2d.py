import numpy as np

from ._ffi import _load_lib_full, _ptr, ffi
from .lossy.quant import dtype_code


# Predictor priors. A name expands to {that predictor, id}; None is unbiased
# over all; "none" is the no-predictor branch alone.
PRIORS = ("planar", "med", "delta_w", "delta_n", "average", "wp_static",
          "delta_1d", "none")

# An OpenZL numeric stream is 1, 2, 4 or 8 bytes per element.
_ELT_WIDTHS = frozenset((1, 2, 4, 8))

# geozl_nodata_mode
_NODATA_NONE, _NODATA_NAN, _NODATA_VALUE = 0, 1, 2


def _nodata_args(arr, nodata):
    """Pick the mode the C side gets. A declared sentinel wins. Otherwise a
    float tile that actually holds NaN gets the automatic path, and everything
    else gets nothing, so a tile with no missing samples never pays for the
    mask stream."""
    if nodata is not None:
        return _NODATA_VALUE, float(nodata)
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        return _NODATA_NAN, 0.0
    return _NODATA_NONE, 0.0


def _is_native(dt):
    return dt.byteorder in ("=", "|") or dt.newbyteorder("=") == dt


def _prepare(tile, width):
    arr = np.ascontiguousarray(tile)
    if not _is_native(arr.dtype):
        raise ValueError(f"dtype {arr.dtype} is not native byte order; the "
                         f"kernels read native, byte-swap it first")
    if width is None:
        if arr.ndim < 2:
            raise ValueError("give width for a 1d tile")
        width = arr.shape[-1]
    elt = arr.dtype.itemsize
    if elt not in _ELT_WIDTHS:
        raise ValueError(f"dtype {arr.dtype} is {elt} bytes per element, an "
                         f"OpenZL numeric stream is 1, 2, 4 or 8")
    return arr, int(width), elt


def compress(tile, *, method, width=None, error=None, nodata=None):
    """Compress a 2d tile into one OpenZL frame through the graph method names.

    method is a recipe, the same string profile puts in its "graph" column, and
    it has to apply to the element width: the transpose and store_lo terminals
    want 2 to 8 bytes. Returns the frame as bytes.

    error is a recipe too. None is lossless. "abs:V" bounds the absolute error,
    "rel:P%" the relative one, and "shot:a=A,b=B,k=K" holds K times the noise of
    a sensor whose variance is A + B*x. Which one fits follows from how the
    measurement error of the data grows with the value, so elevation takes abs,
    optical radiance takes shot, and SAR backscatter takes rel.

    nodata declares the value that marks a sample as never measured, following
    GDAL, where nodata is a property of the band rather than a switch. Left
    unset a float tile still gets its NaN handled, since a NaN says so itself,
    while a sentinel such as -9999 cannot be told from a measurement.
    """
    if not isinstance(method, str) or not method:
        raise ValueError(f"method must be a recipe name, got {method!r}")

    lib = _load_lib_full()
    arr, width, elt = _prepare(tile, width)
    n = arr.size

    # The dtype code goes through even when lossless, the nodata codec needs it
    # to read a sentinel at the right width and signedness.
    code = dtype_code(arr.dtype)
    if error is None:
        code = 0 if code is None else code
    elif code is None:
        raise ValueError(f"quant does not support dtype {arr.dtype}")
    if error is not None and not isinstance(error, str):
        raise ValueError(f"error must be a recipe string such as \"abs:0.5\" "
                         f"or \"rel:1%\", got {error!r}")

    nd_mode, nd_value = _nodata_args(arr, nodata)
    if nd_mode == _NODATA_VALUE and dtype_code(arr.dtype) is None:
        raise ValueError(f"nodata needs a dtype geozl knows, got {arr.dtype}")

    # Sized past the worst case; an incompressible tile still fits in 1.5x.
    cap = 1024 + n * elt + n * elt // 2
    dst = np.empty(cap, np.uint8)
    out_size = ffi.new("size_t*")
    err_ctx = ffi.new("char[]", 256)
    rc = lib.geozl_2d_compress_c(
        method.encode("utf-8"), width,
        ffi.NULL if error is None else error.encode("utf-8"),
        int(code), nd_mode, nd_value,
        _ptr(arr), n, elt, _ptr(dst), cap, out_size, err_ctx, len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.compress failed (method={method!r}): "
                           f"{reason} (ZL error code {rc})")
    return dst[:out_size[0]].tobytes()


def decompress(frame, *, dtype=None, width=None):
    """Decompress a self-describing geozl frame to a numpy array.

    Lossless frames keep the element width but not its sign, so pass dtype to
    recover int vs float. width reshapes to 2d.
    """
    lib = _load_lib_full()
    buf = np.frombuffer(frame, np.uint8)
    dsize = lib.geozl_2d_frame_dsize_c(_ptr(buf), buf.size)
    if dsize == 0:
        raise RuntimeError("geozl.decompress: unreadable frame")

    # Numeric output must be 8-byte aligned; a uint64 backing store guarantees it.
    dsize = int(dsize)
    out = np.empty((dsize + 7) // 8, np.uint64).view(np.uint8)[:dsize]
    out_size = ffi.new("size_t*")
    err_ctx = ffi.new("char[]", 256)
    rc = lib.geozl_2d_decompress_c(_ptr(buf), buf.size, _ptr(out), out.size,
                                   out_size, err_ctx, len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.decompress failed: {reason} "
                           f"(ZL error code {rc})")

    out = out[:out_size[0]]
    if dtype is not None:
        out = out.view(np.dtype(dtype))
    if width:
        out = out.reshape(-1, int(width))
    return out


def _grid_names(method, elt):
    lib = _load_lib_full()
    stride, cap = 48, 64
    names = ffi.new("char[]", stride * cap)
    count = ffi.new("size_t*")
    rc = lib.geozl_2d_grid_c((method or "").encode("utf-8"), elt, names, stride,
                             cap, count)
    if rc != 0:
        raise ValueError(f"method {method!r} is not one of {PRIORS} or None")
    n = min(int(count[0]), cap)
    return [ffi.string(names + i * stride).decode("utf-8") for i in range(n)]


def _order0_bits(arr):
    b = np.frombuffer(arr.tobytes(), np.uint8)
    counts = np.bincount(b, minlength=256).astype(np.float64)
    p = counts[counts > 0] / b.size
    return float(-(p * np.log2(p)).sum())


def profile(tile, *, method="planar", width=None, error=None, reps=5,
            nodata=None):
    """Benchmark every candidate graph on one tile, one row each.

    A diagnostic, not on the compress path. Timing runs in C (checksum off) so
    the numbers are the pure codec. shannon_pct is frame size against the tile's
    order-0 entropy (over 100 means structure was exploited). Every "graph" it
    returns is a method compress takes.

    error and nodata mean the same as they do in compress, and are passed
    through so the graph timed here is the one compress would build. Without it a tile holding
    NaN would be profiled through a different graph than the one it ends up
    compressed with, and the sizes would not line up.
    """
    lib = _load_lib_full()
    arr, width, elt = _prepare(tile, width)
    raw = arr.nbytes
    ideal = raw * _order0_bits(arr) / 8.0
    code = dtype_code(arr.dtype)
    if error is None:
        code = 0 if code is None else code
    elif code is None:
        raise ValueError(f"quant does not support dtype {arr.dtype}")
    err = ffi.NULL if error is None else error.encode("utf-8")

    nd_mode, nd_value = _nodata_args(arr, nodata)
    if nd_mode == _NODATA_VALUE and dtype_code(arr.dtype) is None:
        raise ValueError(f"nodata needs a dtype geozl knows, got {arr.dtype}")

    rows = []
    for name in _grid_names(method, elt):
        comp = ffi.new("size_t*")
        enc = ffi.new("double*")
        dec = ffi.new("double*")
        err_ctx = ffi.new("char[]", 256)
        rc = lib.geozl_2d_bench_c(
            name.encode("utf-8"), width, err, int(code), nd_mode, nd_value,
            _ptr(arr), arr.size, elt, reps, comp, enc, dec, err_ctx,
            len(err_ctx))
        if rc != 0:
            continue
        size = int(comp[0])
        rows.append({
            "graph": name,
            "ratio": raw / size,
            "encode_mbps": raw / enc[0] / 1e6,
            "decode_mbps": raw / dec[0] / 1e6,
            "shannon_pct": 100.0 * ideal / size,
        })
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    return rows