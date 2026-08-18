import math
import re
from decimal import Decimal
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

Error = int | float | str | None

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _nodata_args(arr: np.ndarray, nodata: Any) -> tuple[int, int]:
    """Return the C nodata mode and sentinel bits."""
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


def _zero_recipe(error: str) -> bool:
    """Return whether a simple or canonical recipe declares zero error."""
    forms = (
        ("LINEAR:MAX_ERROR=", ""),
        ("LOG:MAX_ERROR=", "%"),
        ("SQRT:MAX_ERROR=", "N"),
    )
    for prefix, suffix in forms:
        if not error.startswith(prefix) or not error.endswith(suffix):
            continue
        end = len(error) - len(suffix) if suffix else len(error)
        value = error[len(prefix):end]
        if _NUMBER.fullmatch(value):
            return Decimal(value).is_zero()
    return False


def _normalize_error(error: Error) -> str | None:
    """Expand the short Python error syntax to a codec recipe."""
    if error is None:
        return None

    if isinstance(error, (bool, np.bool_)):
        raise ValueError("error must not be a boolean")

    if isinstance(error, (int, np.integer)):
        if error < 0:
            raise ValueError(f"error must not be negative, got {error!r}")
        if error == 0:
            return None
        return f"LINEAR:MAX_ERROR={int(error)}"

    if isinstance(error, (float, np.floating)):
        if not math.isfinite(error):
            raise ValueError(f"error must be finite, got {error!r}")
        if error < 0:
            raise ValueError(f"error must not be negative, got {error!r}")
        if error == 0:
            return None
        return "LINEAR:MAX_ERROR=" + str(error)

    if not isinstance(error, str):
        raise ValueError(f"error must be a number, percentage or recipe, got "
                         f"{error!r}")
    if _zero_recipe(error):
        return None
    if ":" in error:
        return error

    if error.endswith("%"):
        value = error[:-1]
        if not _NUMBER.fullmatch(value):
            raise ValueError(f"relative error must look like \"10%\", got "
                             f"{error!r}")
        amount = Decimal(value)
        if amount.is_zero():
            return None
        if amount < 0 or amount >= 100:
            raise ValueError(f"relative error must be between 0% and 100%, "
                             f"got {error!r}")
        return f"LOG:MAX_ERROR={value}%"

    raise ValueError(f"error must be a number, a percentage such as \"10%\", "
                     f"or a LINEAR, LOG or SQRT recipe, got {error!r}")


def _error_arg(error: str | None) -> Any:
    """Convert a normalized error recipe to the C argument."""
    return ffi.NULL if error is None else error.encode("utf-8")


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


def _prepare(tile: ArrayLike, width: int | None,
             planes: int | None = None) -> tuple[np.ndarray, int, int, int]:
    arr, elt = _as_raster(tile)
    if width is None:
        if arr.ndim < 2:
            raise ValueError("give width for a 1d raster")
        width = arr.shape[-1]
    if planes is None:
        # a (B, Y, X) cube is B stacked images, anything else is one
        planes = arr.shape[0] if arr.ndim >= 3 else 1
    width, planes = int(width), int(planes)
    if planes < 1:
        raise ValueError(f"planes must be at least 1, got {planes}")
    if planes > 1:
        n = arr.size
        if n % planes or (n // planes) % width:
            raise ValueError(
                f"{planes} planes do not split {n} elements into whole rows "
                f"of {width}")
    return arr, width, planes, elt


def _dtype_arg(arr: np.ndarray, error: str | None) -> int:
    """Return the dtype code required by the C API."""
    code = dtype_code(arr.dtype)
    if error is None:
        return 0 if code is None else code
    if code is None:
        raise ValueError(f"the quantizers do not support dtype {arr.dtype}")
    return code


class ProfileResults(list[dict[str, Any]]):
    """Profile rows with the input and benchmark settings used to make them."""

    def __init__(self, *, shape: tuple[int, ...], dtype: np.dtype,
                 raw_bytes: int, width: int, planes: int, prior: str | None,
                 reps: int, error: str | None) -> None:
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.raw_bytes = raw_bytes
        self.width = width
        self.planes = planes
        self.prior = prior
        self.reps = reps
        self.error = error

    def __str__(self) -> str:
        axes = {
            0: "scalar",
            1: "samples",
            2: "rows, columns",
            3: "planes, rows, columns",
        }.get(len(self.shape), "planes, ..., rows, columns")
        plane_word = "plane" if self.planes == 1 else "planes"
        rep_word = "rep" if self.reps == 1 else "reps"
        if self.prior is None:
            predictors = "all predictors"
        elif self.prior == "none":
            predictors = "no predictor"
        else:
            predictors = f"{self.prior} + id"
        mode = self.error or "lossless"

        lines = [
            "input",
            f"  shape    {self.shape}  [{axes}]",
            f"  dtype    {self.dtype}  [{self.dtype.itemsize} bytes/sample]",
            f"  raw      {self.raw_bytes / 1e6:.2f} MB",
            f"  profile  {self.planes} {plane_word} · row width {self.width} · "
            f"{predictors} · {self.reps} {rep_word} · {mode}",
            "",
        ]
        if not self:
            lines.append("no compatible graphs")
            return "\n".join(lines)

        graph_width = 32
        lines.append(
            f"{'graph':{graph_width}} {'ratio':>6} {'enc MB/s':>9} "
            f"{'dec MB/s':>9} {'shan%':>6}")
        for row in self:
            lines.append(
                f"{row['graph']:{graph_width}} {row['ratio']:6.2f} "
                f"{row['encode_mbps']:9.1f} {row['decode_mbps']:9.1f} "
                f"{row['shannon_pct']:6.0f}")
        return "\n".join(lines)


class Graph:
    """A reusable compression graph.

    A graph owns its compression context and is not thread-safe.
    """

    __slots__ = ("_h", "_dtype", "method", "width", "planes", "itemsize",
                 "error", "nodata")

    def __init__(self, handle: Any, *, method: str, width: int, planes: int,
                 dtype: np.dtype, itemsize: int, error: str | None,
                 nodata: Any) -> None:
        self._h = handle
        self.method = method
        self.width = width
        self.planes = planes
        self._dtype = dtype
        self.itemsize = itemsize
        self.error = error
        self.nodata = nodata

    @property
    def dtype(self) -> np.dtype:
        return self._dtype


def graph(raster: ArrayLike, method: str, *, width: int | None = None,
          planes: int | None = None, error: Error = None,
          nodata: Any = None) -> Graph:
    """Build a reusable graph for ``method``.

    ``width`` defaults to the last axis. ``planes`` defaults to the first axis
    for a ``(B, Y, X)`` array and to 1 otherwise. Row-based predictors reset at
    each plane; use ``width=Y*X, planes=1`` to predict between bands.

    ``error=None`` or zero is lossless. A positive number selects ``LINEAR``;
    a percentage string such as ``"1%"`` selects ``LOG``. Full ``LINEAR``,
    ``LOG`` and ``SQRT`` recipes are also accepted. The raster fixes the lossy
    domain checked by ``compress``; use data spanning the product. ``nodata``
    sets a sentinel; NaNs are detected when it is omitted for floating-point
    rasters.
    """
    if not isinstance(method, str) or not method:
        raise ValueError(f"method must be a recipe name, got {method!r}")
    error_recipe = _normalize_error(error)
    err = _error_arg(error_recipe)

    lib = _load_lib_full()
    arr, width, planes, elt = _prepare(raster, width, planes)
    code = _dtype_arg(arr, error_recipe)
    nd_mode, nd_bits = _nodata_args(arr, nodata)

    out = ffi.new("geozl_2d_graph**")
    err_ctx = ffi.new("char[]", 256)
    rc = lib.geozl_2d_graph_open_c(
        out, method.encode("utf-8"), width, planes, err, int(code), nd_mode, nd_bits,
        _ptr(arr), arr.size, elt, err_ctx, len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.graph failed (method={method!r}): "
                           f"{reason} (ZL error code {rc})")

    return Graph(ffi.gc(out[0], lib.geozl_2d_graph_close_c),
                 method=method, width=width, planes=planes,
                 dtype=arr.dtype, itemsize=elt,
                 error=error_recipe, nodata=nodata)


def compress(tile: ArrayLike, *, graph: Graph) -> bytes:
    """Compress ``tile`` with a prepared graph and return the frame bytes.

    The dtype must match exactly. Geometry is not checked against the graph. A
    lossy tile outside the domain used to build the graph is refused.
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"graph must be a geozl.Graph, got "
                        f"{type(graph).__name__}")
    lib = _load_lib_full()
    arr, elt = _as_raster(tile)
    if arr.dtype != graph.dtype:
        raise ValueError(f"this graph was built for dtype {graph.dtype}, "
                         f"got {arr.dtype}")
    n = arr.size

    # Sized past the worst case; an incompressible tile still fits in 1.5x.
    cap = 1024 + n * elt + n * elt // 2
    dst = np.empty(cap, np.uint8)
    out_size = ffi.new("size_t*")
    err_ctx = ffi.new("char[]", 256)
    rc = lib.geozl_2d_compress_graph_c(
        graph._h, _ptr(arr), n, _ptr(dst), cap, out_size, err_ctx, len(err_ctx))
    if rc != 0:
        reason = ffi.string(err_ctx).decode("utf-8", "replace")
        raise RuntimeError(f"geozl.compress failed (method={graph.method!r}): "
                           f"{reason} (ZL error code {rc})")
    return dst[:out_size[0]].tobytes()


def decompress(frame: bytes, *, verify: bool = True,
               max_output_size: int | None = None) -> np.ndarray:
    """Decompress a frame into a flat ``uint8`` array.

    Restore the dtype and shape with ``.view(dtype).reshape(shape)``. Set
    ``verify=False`` to skip checksum verification.

    Use ``max_output_size`` to limit allocations when reading untrusted frames.
    """
    lib = _load_lib_full()
    buf = np.frombuffer(frame, np.uint8)
    dsize = lib.geozl_2d_frame_dsize_c(_ptr(buf), buf.size)
    if dsize == 0:
        raise RuntimeError("geozl.decompress: unreadable frame")

    # Numeric output must be 8-byte aligned; a uint64 backing store guarantees it.
    dsize = int(dsize)
    if max_output_size is not None and dsize > max_output_size:
        raise ValueError(
            f"geozl.decompress: the frame declares {dsize} bytes of output, "
            f"above the {max_output_size} allowed"
        )
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
    """Return MB/s, or infinity when the timer reports no elapsed time."""
    return math.inf if seconds <= 0.0 else nbytes / seconds / 1e6


def _order0_bits(arr: np.ndarray) -> float:
    b = np.frombuffer(arr.tobytes(), np.uint8)
    counts = np.bincount(b, minlength=256).astype(np.float64)
    p = counts[counts > 0] / b.size
    return float(-(p * np.log2(p)).sum())


def profile(tile: ArrayLike, *, prior: str | None = "planar",
            width: int | None = None, planes: int | None = None,
            error: Error = None,
            reps: int = 5, nodata: Any = None,
            verify: bool = False) -> ProfileResults:
    """Benchmark candidate graphs on ``tile``, sorted by compression ratio.

    ``prior`` selects one predictor plus the identity path. Use ``None`` for all
    predictors or ``"none"`` for the no-predictor path. Other configuration
    arguments match ``graph``.

    Each row contains ``graph``, ``bytes``, ``ratio``, ``encode_mbps``,
    ``decode_mbps`` and ``shannon_pct``. ``verify`` controls checksum checks
    during decode timing; generated frames still include checksums.
    """
    if reps < 1:
        raise ValueError(f"reps must be at least 1, got {reps}")
    error_recipe = _normalize_error(error)
    err = _error_arg(error_recipe)
    lib = _load_lib_full()
    arr, width, planes, elt = _prepare(tile, width, planes)
    raw = arr.nbytes
    ideal = raw * _order0_bits(arr) / 8.0
    code = _dtype_arg(arr, error_recipe)
    nd_mode, nd_bits = _nodata_args(arr, nodata)

    rows = ProfileResults(
        shape=arr.shape, dtype=arr.dtype, raw_bytes=raw, width=width,
        planes=planes, prior=prior, reps=reps, error=error_recipe)
    for name in _grid_names(prior, elt):
        comp = ffi.new("size_t*")
        enc = ffi.new("double*")
        dec = ffi.new("double*")
        err_ctx = ffi.new("char[]", 256)
        # checksum on, so "bytes" is the frame compress writes and not an
        # estimate of it
        rc = lib.geozl_2d_bench_c(
            name.encode("utf-8"), width, planes, err, int(code), nd_mode, nd_bits,
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
