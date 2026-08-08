import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._dtype import dtype_code
from ._ffi import _load_lib_full, _ptr, ffi
from .lossless.nodata import nodata_bits

# Predictor priors. A name expands to {that predictor, id}; None is unbiased
# over all; "none" is the no-predictor branch alone.
PRIORS = ("planar", "med", "delta_w", "delta_n", "average", "wp_static",
          "delta_1d", "none")

# An OpenZL numeric stream is 1, 2, 4 or 8 bytes per element.
_ELT_WIDTHS = frozenset((1, 2, 4, 8))

# geozl_nodata_mode
_NODATA_NONE, _NODATA_NAN, _NODATA_VALUE = 0, 1, 2


def _nodata_args(arr: np.ndarray, nodata: Any) -> tuple[int, int]:
    """Mode and sentinel bits for the C side. A declared sentinel wins,
    otherwise a float raster holding NaN takes the automatic path and everything
    else gets nothing, so a raster with no missing samples pays for no mask."""
    if nodata is None:
        if arr.dtype.kind == "f" and np.isnan(arr).any():
            return _NODATA_NAN, 0
        return _NODATA_NONE, 0
    if dtype_code(arr.dtype) is None:
        raise ValueError(f"nodata needs a dtype geozl knows, got {arr.dtype}")
    # as bits a NaN matches one payload, not every NaN in the raster
    if arr.dtype.kind == "f" and isinstance(nodata, float) and math.isnan(nodata):
        return _NODATA_NAN, 0
    return _NODATA_VALUE, nodata_bits(nodata, arr.dtype)


def _error_arg(error: str | None) -> Any:
    """The error recipe as C wants it. A recipe rather than a number so it
    crosses graph, profile and bench unchanged, and so the string names which
    quantizer runs. A bare float is refused rather than read as an absolute
    bound, since there are three curves and a number says nothing about which
    one was meant."""
    if error is None:
        return ffi.NULL
    if not isinstance(error, str):
        raise ValueError(f"error must be a recipe string, one of "
                         f"\"LINEAR:MAX_ERROR=V\", \"LOG:MAX_ERROR=P%\" or "
                         f"\"SQRT:MAX_ERROR=VN\", got {error!r}")
    return error.encode("utf-8")


def _is_native(dt: np.dtype) -> bool:
    return dt.byteorder in ("=", "|") or dt.newbyteorder("=") == dt


def _as_raster(tile: ArrayLike) -> tuple[np.ndarray, int]:
    arr = np.ascontiguousarray(tile)
    if not _is_native(arr.dtype):
        raise ValueError(f"dtype {arr.dtype} is not native byte order; the "
                         f"kernels read native, byte-swap it first")
    elt = arr.dtype.itemsize
    if elt not in _ELT_WIDTHS:
        raise ValueError(f"dtype {arr.dtype} is {elt} bytes per element, an "
                         f"OpenZL numeric stream is 1, 2, 4 or 8")
    return arr, elt


def _prepare(tile: ArrayLike,
             width: int | None) -> tuple[np.ndarray, int, int]:
    arr, elt = _as_raster(tile)
    if width is None:
        if arr.ndim < 2:
            raise ValueError("give width for a 1d raster")
        width = arr.shape[-1]
    return arr, int(width), elt


def _dtype_arg(arr: np.ndarray, error: str | None) -> int:
    """The dtype code goes through even when lossless, the nodata codec needs it
    to read a sentinel at the right width and signedness."""
    code = dtype_code(arr.dtype)
    if error is None:
        return 0 if code is None else code
    if code is None:
        raise ValueError(f"the quantizers do not support dtype {arr.dtype}")
    return code


class Graph:
    """A built graph, reusable across tiles. Freed when it is collected.

    Not thread safe: the compression context is bound to the graph, so one
    Graph per thread.
    """

    __slots__ = ("_h", "method", "width", "dtype", "itemsize", "error",
                 "nodata")

    def __init__(self, handle: Any, *, method: str, width: int,
                 dtype: np.dtype, itemsize: int, error: str | None,
                 nodata: Any) -> None:
        self._h = handle
        self.method = method
        self.width = width
        self.dtype = dtype
        self.itemsize = itemsize
        self.error = error
        self.nodata = nodata


def graph(raster: ArrayLike, method: str, *, width: int | None = None,
          error: str | None = None, nodata: Any = None) -> Graph:
    """Build one graph, to compress many tiles through.

    method is a recipe, the same string profile puts in its "graph" column.

    width is the stride the predictor steps by, taken from the last axis when
    left unset. Not the row length: a width of Y*X over a (B, Y, X) cube
    predicts each band from the one before it.

    error is a recipe. None is lossless. "LINEAR:MAX_ERROR=V" bounds the
    absolute error, "LOG:MAX_ERROR=P%" the relative one, and
    "SQRT:MAX_ERROR=VN" holds V times the noise of a sensor whose variance is
    a + b*x.

    nodata is the value that marks a sample as never measured. Left unset, a
    float raster holding NaN takes NaN as its own.
    """
    if not isinstance(method, str) or not method:
        raise ValueError(f"method must be a recipe name, got {method!r}")
    err = _error_arg(error)

    lib = _load_lib_full()
    arr, width, elt = _prepare(raster, width)
    code = _dtype_arg(arr, error)
    nd_mode, nd_bits = _nodata_args(arr, nodata)

    out = ffi.new("geozl_2d_graph**")
    err_ctx = ffi.new("char[]", 256)
    rc = lib.geozl_2d_graph_open_c(
        out, method.encode("utf-8"), width, err, int(code), nd_mode, nd_bits,
        _ptr(arr), arr.size, elt, err_ctx, len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.graph failed (method={method!r}): "
                           f"{reason} (ZL error code {rc})")

    return Graph(ffi.gc(out[0], lib.geozl_2d_graph_close_c),
                 method=method, width=width, dtype=arr.dtype, itemsize=elt,
                 error=error, nodata=nodata)


def compress(tile: ArrayLike, *, graph: Graph) -> bytes:
    """Compress one tile through a prepared graph. Returns the frame as bytes.

    Only the element width is checked. A tile whose rows do not match the
    graph's stride still compresses, and predicts across a boundary that is not
    there.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"graph must be a geozl.Graph, got "
                        f"{type(graph).__name__}")
    lib = _load_lib_full()
    arr, elt = _as_raster(tile)
    if elt != graph.itemsize:
        raise ValueError(f"this graph streams {graph.itemsize}-byte elements, "
                         f"got {arr.dtype}")
    n = arr.size

    # Sized past the worst case; an incompressible tile still fits in 1.5x.
    cap = 1024 + n * elt + n * elt // 2
    dst = np.empty(cap, np.uint8)
    out_size = ffi.new("size_t*")
    err_ctx = ffi.new("char[]", 256)
    rc = lib.geozl_2d_compress_graph_c(graph._h, _ptr(arr), n, _ptr(dst), cap,
                                       out_size, err_ctx, len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.compress failed (method={graph.method!r}): "
                           f"{reason} (ZL error code {rc})")
    return dst[:out_size[0]].tobytes()


def decompress(frame: bytes, *, verify: bool = True) -> np.ndarray:
    """Decompress a self-describing geozl frame into a flat uint8 array.

    The frame names its own codecs but not the caller's type or shape, so the
    caller finishes with .view(dtype).reshape(shape).

    verify checks the checksums the frame carries, which is what tells a frame
    that rotted in storage from one that did not. Off buys 1 to 30 per cent of
    decode and gives up the only warning you get.
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
                                   out_size, int(verify), err_ctx,
                                   len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.decompress failed: {reason} "
                           f"(ZL error code {rc})")
    return out[:out_size[0]]


def _grid_names(prior: str | None, elt: int) -> list[str]:
    lib = _load_lib_full()
    stride, cap = 48, 64
    names = ffi.new("char[]", stride * cap)
    count = ffi.new("size_t*")
    rc = lib.geozl_2d_grid_c((prior or "").encode("utf-8"), elt, names, stride,
                             cap, count)
    if rc != 0:
        raise ValueError(f"prior {prior!r} is not one of {PRIORS} or None")
    n = min(int(count[0]), cap)
    return [ffi.string(names + i * stride).decode("utf-8") for i in range(n)]


def _mbps(nbytes: int, seconds: float) -> float:
    """inf when the whole run stayed inside a clock tick, which a small tile on
    a coarse clock does. A number there would be one tick wide."""
    return math.inf if seconds <= 0.0 else nbytes / seconds / 1e6


def _order0_bits(arr: np.ndarray) -> float:
    b = np.frombuffer(arr.tobytes(), np.uint8)
    counts = np.bincount(b, minlength=256).astype(np.float64)
    p = counts[counts > 0] / b.size
    return float(-(p * np.log2(p)).sum())


def profile(tile: ArrayLike, *, prior: str | None = "planar",
            width: int | None = None, error: str | None = None,
            reps: int = 5, nodata: Any = None,
            verify: bool = False) -> list[dict[str, Any]]:
    """Benchmark every candidate graph on one tile, one row each, ranked by
    ratio. A diagnostic, not on the compress path.

    Timing is one compressor and one decoder per graph, so it reports the codec
    and not the build. "bytes" is the frame compress writes, "graph" is a method
    geozl.graph takes, and shannon_pct is that frame against the tile's order-0
    entropy, over 100 where structure was exploited. A throughput is inf when
    the tile was too small for the clock.

    prior narrows the sweep to one predictor plus the id pass. None sweeps them
    all, "none" sweeps the no-predictor branch alone.

    width, error and nodata mean what they do in graph, and belong here too, or
    the ranking is of graphs nobody will build.

    verify defaults off, unlike compress, so decode timing skips a constant
    every graph pays alike.
    """
    if reps < 1:
        raise ValueError(f"reps must be at least 1, got {reps}")
    err = _error_arg(error)
    lib = _load_lib_full()
    arr, width, elt = _prepare(tile, width)
    raw = arr.nbytes
    ideal = raw * _order0_bits(arr) / 8.0
    code = _dtype_arg(arr, error)
    nd_mode, nd_bits = _nodata_args(arr, nodata)

    rows: list[dict[str, Any]] = []
    for name in _grid_names(prior, elt):
        comp = ffi.new("size_t*")
        enc = ffi.new("double*")
        dec = ffi.new("double*")
        err_ctx = ffi.new("char[]", 256)
        # checksum on, so "bytes" is the frame compress writes and not an
        # estimate of it
        rc = lib.geozl_2d_bench_c(
            name.encode("utf-8"), width, err, int(code), nd_mode, nd_bits,
            _ptr(arr), arr.size, elt, reps, 1, int(verify), comp, enc, dec,
            err_ctx, len(err_ctx))
        if rc != 0:
            continue
        size = int(comp[0])
        rows.append({
            "graph": name,
            "bytes": size,
            "ratio": raw / size,
            "encode_mbps": _mbps(raw, enc[0]),
            "decode_mbps": _mbps(raw, dec[0]),
            "shannon_pct": 100.0 * ideal / size,
        })
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    return rows
