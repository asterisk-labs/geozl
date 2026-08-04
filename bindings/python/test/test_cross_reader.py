"""Every codec is written twice, as encode_X_binding.c and in Python, and both
sides pack the same wire format. This is the guard that they agree. A frame from
one side reads on the other, byte for byte on a lossy frame.
"""
import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

_W = 32

_PREDICTORS = [
    ("planar", geozl.lossless.Planar),
    ("delta_w", geozl.lossless.DeltaW),
    ("delta_n", geozl.lossless.DeltaN),
    ("med", geozl.lossless.Med),
    ("average", geozl.lossless.Average),
    ("wp_static", geozl.lossless.WpStatic),
]

_DTYPES = ["int16", "uint16", "int32", "uint32", "int8", "uint8"]


def _tile(dtype="int16", shape=(_W, _W)):
    rng = np.random.default_rng(0)
    y, x = np.indices(shape)
    return ((x * 3 + y * 5) % 100 + rng.integers(0, 4, shape)).astype(dtype)


def _method(predictor, dtype):
    # transpose wants 2 to 8 bytes per element
    tail = "zigzag>entropy" if np.dtype(dtype).itemsize == 1 \
        else "zigzag>transpose>entropy"
    return f"{predictor}>{tail}"


def _read_with_python_decoders(frame):
    d = zl.DCtx()
    geozl.register_decoders(d)
    return d.decompress(frame)[0].content.as_nparray()


def _frame_from_python_node(node, arr):
    c = zl.Compressor()
    c.select_starting_graph(node(c, zl.graphs.Compress()(c)))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    return bytes(cc.compress(
        [zl.Input(zl.Type.Numeric, np.ascontiguousarray(arr).reshape(-1))]))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("predictor,node", _PREDICTORS,
                         ids=[p for p, _ in _PREDICTORS])
def test_a_c_frame_reads_in_the_python_decoders(predictor, node, dtype):
    arr = _tile(dtype)
    frame = geozl.compress(arr, method=_method(predictor, dtype))
    out = _read_with_python_decoders(frame).view(arr.dtype)
    assert np.array_equal(out.reshape(arr.shape), arr)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("predictor,node", _PREDICTORS,
                         ids=[p for p, _ in _PREDICTORS])
def test_a_python_frame_reads_in_the_c_decoders(predictor, node, dtype):
    arr = _tile(dtype)
    frame = _frame_from_python_node(node(_W), arr)
    assert np.array_equal(geozl.decompress(frame, dtype=dtype, width=_W), arr)


@pytest.mark.parametrize("error", [
    "LINEAR:MAX_ERROR=2",
    "LINEAR:MAX_ERROR=0.5",
    "LOG:MAX_ERROR=1%",
    "SQRT:MAX_ERROR=2N",
])
def test_both_decoders_return_the_same_bytes_on_a_lossy_frame(error):
    """The quantiser runs on encode, so a decoder that reconstructs differently
    read the header differently. A tolerance would hide it."""
    # SQRT fits its curve from the raster, which needs a few hundred blocks
    w = 256
    rng = np.random.default_rng(1)
    n = w * w
    arr = (10.0 ** np.linspace(-3, 4, n) * rng.uniform(0.9, 1.1, n)
           ).reshape(w, w).astype(np.float32)
    frame = geozl.compress(arr, method="planar>zigzag>transpose>entropy",
                           error=error)
    from_c = geozl.decompress(frame, dtype="float32", width=w)
    from_py = _read_with_python_decoders(frame).view(np.float32)
    assert np.array_equal(from_py.reshape(from_c.shape).view(np.uint32),
                          from_c.view(np.uint32))


@pytest.mark.parametrize("dtype,hole", [
    ("int32", -9999),
    ("int64", 2 ** 53 + 1),
    ("uint64", 2 ** 64 - 1),
    ("float32", np.nan),
])
def test_the_nodata_header_reads_in_both_directions(dtype, hole):
    arr = _tile("int32").astype(dtype)
    arr[4:12, 6:20] = np.array(hole).astype(dtype)
    kw = {} if dtype == "float32" else {"nodata": hole}
    frame = geozl.compress(arr, method="planar>zigzag>transpose>entropy", **kw)

    from_c = geozl.decompress(frame, dtype=dtype, width=_W)
    from_py = _read_with_python_decoders(frame).view(np.dtype(dtype))
    assert np.array_equal(from_py.reshape(arr.shape).view(f"u{arr.itemsize}"),
                          from_c.view(f"u{arr.itemsize}"))
