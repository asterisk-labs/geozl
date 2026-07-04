import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_CHECKSUM_DISABLE = 2   # OpenZL ternary: disable


def make_tile(shape, dtype, pattern):
    """A tile built on the fly. random covers every byte value and the integer
    wraparound, gradient is what a predictor predicts well, constant is the all
    zero residual edge."""
    dt = np.dtype(dtype)
    n = shape[0] * shape[1]
    if pattern == "constant":
        return np.full(shape, 7, dtype=dt)
    if pattern == "gradient":
        if np.issubdtype(dt, np.integer):
            hi = min(int(np.iinfo(dt).max), n)
            return (np.arange(n) % (hi + 1)).astype(dt).reshape(shape)
        return np.linspace(0.0, 1000.0, n).astype(dt).reshape(shape)
    rng = np.random.default_rng(abs(hash((shape, dt.str))) % (2**32))
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=dt)
    return (rng.random(shape) * 1000.0).astype(dt)


def roundtrip(node, arr, *, disable_checksum=False):
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = node(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    if disable_checksum:
        cc.set_parameter(zl.CParam.ContentChecksum, _CHECKSUM_DISABLE)
    flat = np.ascontiguousarray(arr).reshape(-1)
    frame = bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))
    d = zl.DCtx()
    geozl.lossless.register_decoders(d)
    geozl.lossy.register_decoders(d)
    # as_nparray types the stream by width alone, unsigned int, so reinterpret
    # it to the tile dtype the decoded bytes actually hold.
    out = d.decompress(frame)[0].content.as_nparray()
    return out.view(arr.dtype)


# native OpenZL numeric types, every element width 1/2/4/8, integer and float
_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64,
           np.int8, np.int16, np.int32, np.int64,
           np.float16, np.float32, np.float64]

# non-square, single row and single column exercise the row boundary logic
_SHAPES = [(16, 16), (7, 13), (1, 32), (32, 1), (4, 4)]
_PATTERNS = ["random", "gradient", "constant"]

_PREDICTORS = [geozl.lossless.DeltaWInt, geozl.lossless.DeltaNInt,
               geozl.lossless.PlanarInt, geozl.lossless.MedInt]


@pytest.mark.parametrize("node", _PREDICTORS, ids=lambda n: n.__name__)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_predictor_bit_exact(node, dtype, shape, pattern):
    arr = make_tile(shape, dtype, pattern)
    out = roundtrip(node(shape[1]), arr)
    assert np.array_equal(out, arr.reshape(-1))


_LOSSY_DTYPES = [np.uint8, np.uint16, np.int16, np.float32, np.float64]


@pytest.mark.parametrize("dtype", _LOSSY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("max_error", [1, 5, 50])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", ["random", "gradient"])
def test_quant_linear_bound(dtype, max_error, shape, pattern):
    arr = make_tile(shape, dtype, pattern)
    node = geozl.lossy.quant_linear(max_error, dtype)
    out = roundtrip(node, arr, disable_checksum=True)
    err = np.abs(out.astype(np.float64) - arr.reshape(-1).astype(np.float64))
    assert err.max() <= max_error


def test_quant_linear_rejects_content_checksum():
    arr = make_tile((16, 16), np.uint16, "random")
    node = geozl.lossy.quant_linear(5, np.uint16)
    with pytest.raises(Exception):
        roundtrip(node, arr, disable_checksum=False)


def test_quant_linear_rejects_bad_max_error():
    with pytest.raises(ValueError):
        geozl.lossy.quant_linear(0, np.uint16)