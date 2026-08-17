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
_B = 4

_PREDICTORS = [
    ("planar", geozl.lossless.Planar),
    ("delta_w", geozl.lossless.DeltaW),
    ("delta_n", geozl.lossless.DeltaN),
    ("med", geozl.lossless.Med),
    ("average", geozl.lossless.Average),
    ("wp_static", geozl.lossless.WpStatic),
]

# delta_w only reads to its left, so it carries no plane count on either side.
_STACKED = [(p, n) for p, n in _PREDICTORS if p != "delta_w"]

_DTYPES = ["int16", "uint16", "int32", "uint32", "int8", "uint8"]


def _tile(dtype="int16", shape=(_W, _W)):
    rng = np.random.default_rng(0)
    y, x = np.indices(shape)
    return ((x * 3 + y * 5) % 100 + rng.integers(0, 4, shape)).astype(dtype)


def _cube(dtype="int16"):
    return np.stack([_tile(dtype) + b for b in range(_B)]).astype(dtype)


def _method(predictor, dtype):
    # transpose wants 2 to 8 bytes per element
    tail = "zigzag>entropy" if np.dtype(dtype).itemsize == 1 \
        else "zigzag>transpose>entropy"
    return f"{predictor}>{tail}"


def _frame_from_c(arr, method, **kw):
    return geozl.compress(arr, graph=geozl.graph(arr, method=method, **kw))


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
    frame = _frame_from_c(arr, _method(predictor, dtype))
    out = _read_with_python_decoders(frame).view(arr.dtype)
    assert np.array_equal(out.reshape(arr.shape), arr)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("predictor,node", _PREDICTORS,
                         ids=[p for p, _ in _PREDICTORS])
def test_a_python_frame_reads_in_the_c_decoders(predictor, node, dtype):
    arr = _tile(dtype)
    frame = _frame_from_python_node(node(_W), arr)
    out = geozl.decompress(frame).view(dtype).reshape(-1, _W)
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("predictor,node", _STACKED,
                         ids=[p for p, _ in _STACKED])
def test_a_stacked_c_frame_reads_in_the_python_decoders(predictor, node):
    arr = _cube()
    frame = _frame_from_c(arr, _method(predictor, "int16"))
    out = _read_with_python_decoders(frame).view(arr.dtype)
    assert np.array_equal(out.reshape(arr.shape), arr)


@pytest.mark.parametrize("predictor,node", _STACKED,
                         ids=[p for p, _ in _STACKED])
def test_a_stacked_python_frame_reads_in_the_c_decoders(predictor, node):
    arr = _cube()
    frame = _frame_from_python_node(node(_W, planes=_B), arr)
    out = geozl.decompress(frame).view(arr.dtype)
    assert np.array_equal(out.reshape(arr.shape), arr)


@pytest.mark.parametrize("predictor,node", _STACKED,
                         ids=[p for p, _ in _STACKED])
def test_planes_change_the_python_frame(predictor, node):
    arr = _cube()
    assert (_frame_from_python_node(node(_W, planes=_B), arr)
            != _frame_from_python_node(node(_W), arr))


# pfor is a terminal, so test it separately from the predictors above.
_PFOR_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32",
                "int64", "uint64"]


@pytest.mark.parametrize("dtype", _PFOR_DTYPES)
@pytest.mark.parametrize("shape", [(_W, _W), (1, 129), (13, 17)],
                         ids=["square", "tail", "ragged"])
def test_a_c_pfor_frame_reads_in_the_python_decoders(dtype, shape):
    arr = _tile(dtype, shape)
    frame = _frame_from_c(arr, "planar>zigzag>pfor", width=shape[1])
    out = _read_with_python_decoders(frame).view(arr.dtype)
    assert np.array_equal(out.reshape(arr.shape), arr)


def _pfor_frame_from_python(arr):
    c = zl.Compressor()
    c.select_starting_graph(geozl.lossless.Pfor()(c, zl.graphs.Store()))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    return bytes(cc.compress(
        [zl.Input(zl.Type.Numeric, np.ascontiguousarray(arr).reshape(-1))]))


@pytest.mark.parametrize("dtype", _PFOR_DTYPES)
def test_a_python_pfor_frame_reads_in_the_c_decoders(dtype):
    arr = _tile(dtype)
    out = geozl.decompress(_pfor_frame_from_python(arr))
    assert np.array_equal(out.view(dtype).reshape(-1, _W), arr)


@pytest.mark.parametrize("n", [1, 255, 256, 257, 512, 4096])
def test_pfor_decodes_a_tile_that_packs_to_the_minimum(n):
    """The allocation guard accepts the minimum two-byte block encoding."""
    arr = np.zeros(n, dtype="uint16")
    frame = geozl.compress(arr, graph=geozl.graph(arr, "id>pfor", width=n))
    assert geozl.decompress(frame).view("uint16").tobytes() == arr.tobytes()


def test_c_and_python_pfor_bindings_write_the_same_bytes():
    from geozl.lossless import pfor as _pfor
    for dtype in ("uint8", "uint16", "uint32", "uint64"):
        arr = _tile(dtype).reshape(-1)
        frame = _pfor_frame_from_python(arr)
        assert _pfor.encode(arr) in frame


def test_delta_w_takes_no_plane_count():
    with pytest.raises(ValueError, match="does not support planes"):
        geozl.lossless.DeltaW(_W, planes=_B)


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
    frame = _frame_from_c(arr, "planar>zigzag>transpose>entropy", error=error)
    from_c = geozl.decompress(frame).view(np.float32).reshape(w, w)
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
    frame = _frame_from_c(arr, "planar>zigzag>transpose>entropy", **kw)

    from_c = geozl.decompress(frame).view(np.dtype(dtype)).reshape(_W, _W)
    from_py = _read_with_python_decoders(frame).view(np.dtype(dtype))
    assert np.array_equal(from_py.reshape(arr.shape).view(f"u{arr.itemsize}"),
                          from_c.view(f"u{arr.itemsize}"))
