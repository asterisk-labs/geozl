import zlib

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_CHECKSUM_DISABLE = 2   # OpenZL ternary parameter, disable


def make_tile(shape, dtype, pattern):
    """A tile built on the fly."""
    dt = np.dtype(dtype)
    n = shape[0] * shape[1]
    if pattern == "constant":
        return np.full(shape, 7, dtype=dt)
    if pattern == "gradient":
        if np.issubdtype(dt, np.integer):
            hi = min(int(np.iinfo(dt).max), n)
            return (np.arange(n) % (hi + 1)).astype(dt).reshape(shape)
        return np.linspace(0.0, 1000.0, n).astype(dt).reshape(shape)
    seed = zlib.crc32(f"{shape}-{dt.str}".encode())
    rng = np.random.default_rng(seed)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=dt)
    return (rng.random(shape) * 1000.0).astype(dt)


def compress(node, arr, *, disable_checksum=False):
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
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def roundtrip(node, arr, *, disable_checksum=False):
    frame = compress(node, arr, disable_checksum=disable_checksum)
    d = zl.DCtx()
    geozl.register_decoders(d)
    # as_nparray types the stream by width alone, unsigned int, so reinterpret it
    # to the tile dtype the decoded bytes actually hold.
    out = d.decompress(frame)[0].content.as_nparray()
    return out.view(arr.dtype)


# native OpenZL numeric types, every element width 1/2/4/8, integer and float
_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64,
           np.int8, np.int16, np.int32, np.int64,
           np.float16, np.float32, np.float64]

# non-square, single row and single column exercise the row boundary logic
_SHAPES = [(16, 16), (7, 13), (1, 32), (32, 1), (4, 4)]
_PATTERNS = ["random", "gradient", "constant"]

_PREDICTORS = [geozl.lossless.DeltaW, geozl.lossless.DeltaN,
               geozl.lossless.Planar, geozl.lossless.Med,
               geozl.lossless.Average, geozl.lossless.WpStatic]


@pytest.mark.parametrize("node", _PREDICTORS, ids=lambda n: n.__name__)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_predictor_bit_exact(node, dtype, shape, pattern):
    arr = make_tile(shape, dtype, pattern)
    out = roundtrip(node(shape[1]), arr)
    assert np.array_equal(out, arr.reshape(-1))


def test_predictor_decoder_actually_runs():
    # a static graph forces the node, but confirm the decoder fires so that a
    # frame stored raw could never pass the round trip without touching the codec
    fired = []

    class Spy(geozl.lossless.PlanarDecoder):
        def decode(self, state):
            fired.append(1)
            super().decode(state)

    arr = make_tile((32, 32), np.uint16, "gradient")
    frame = compress(geozl.lossless.Planar(32), arr)
    d = zl.DCtx()
    d.register_custom_decoder(Spy())
    d.decompress(frame)
    assert fired


_LOSSY_DTYPES = [np.uint8, np.uint16, np.int16, np.float32, np.float64]


@pytest.mark.parametrize("dtype", _LOSSY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("error", [0.25, 1, 5, 50])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", ["random", "gradient"])
def test_quant_bound(dtype, error, shape, pattern):
    arr = make_tile(shape, dtype, pattern)
    node = geozl.lossy.Quant(f"abs:{error}", arr)
    out = roundtrip(node, arr, disable_checksum=True)
    err = np.abs(out.astype(np.float64) - arr.reshape(-1).astype(np.float64))
    assert err.max() <= error


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int8, np.int16],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("error", [0.25, 1, 5, 50])
def test_quant_bound_at_type_extremes(dtype, error):
    # the min and max of the type, where reconstruction clamps rather than wraps.
    # the bound must still hold, a clamp that overshoots is a real bug
    info = np.iinfo(dtype)
    arr = np.array([[info.min, info.max, info.min, info.max]], dtype=dtype)
    node = geozl.lossy.Quant(f"abs:{error}", arr)
    out = roundtrip(node, arr, disable_checksum=True)
    err = np.abs(out.astype(np.float64) - arr.reshape(-1).astype(np.float64))
    assert err.max() <= error


def test_quant_rejects_content_checksum():
    arr = make_tile((16, 16), np.uint16, "random")
    node = geozl.lossy.Quant("abs:5", arr)
    with pytest.raises(Exception):
        roundtrip(node, arr, disable_checksum=False)


def test_quant_rejects_bad_error():
    with pytest.raises(ValueError):
        geozl.lossy.Quant("abs:0", np.arange(64, dtype=np.uint16))


# The relative bound is the one a fixed step cannot hold, since the tolerance
# has to follow the value across the whole range.
@pytest.mark.parametrize("pct", [0.5, 1.0, 10.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64],
                         ids=lambda d: np.dtype(d).name)
def test_quant_relative_bound(pct, dtype):
    arr = (10.0 ** np.linspace(-9, 3, 4096)).astype(dtype).reshape(64, 64)
    node = geozl.lossy.Quant(f"rel:{pct}%", arr)
    out = roundtrip(node, arr, disable_checksum=True)
    x = arr.reshape(-1).astype(np.float64)
    err = np.abs(out.astype(np.float64) - x)
    assert (err <= (pct / 100.0) * np.abs(x)).all()


# Zero has no relative neighbourhood, so the bound is only satisfied by giving
# it back exactly. Same for the subnormals no grid can address.
@pytest.mark.parametrize("dtype", [np.float32, np.float64],
                         ids=lambda d: np.dtype(d).name)
def test_quant_relative_preserves_zero_exactly(dtype):
    arr = np.zeros(4096, dtype=dtype).reshape(64, 64)
    arr[0, 1:] = (10.0 ** np.linspace(-6, 2, 63)).astype(dtype)
    node = geozl.lossy.Quant("rel:1%", arr)
    out = roundtrip(node, arr, disable_checksum=True)
    assert (out.reshape(arr.shape)[1:] == 0).all()


# The shot curve holds k times the noise of a sensor whose variance is a + b*x,
# so the tolerance widens with the square root of the signal. This one sits
# within a few parts per million of its bound and holds only because the
# encoder picks the nearest reconstruction rather than rounding in the warped
# domain: the curve is convex, so at the midpoint of a step the level above is
# further away than the one below, and plain rounding would overshoot.
def test_quant_shot_bound():
    arr = np.linspace(0.0, 10000.0, 4096, dtype=np.float32).reshape(64, 64)
    node = geozl.lossy.Quant("shot:a=4,b=1,k=0.5", arr)
    out = roundtrip(node, arr, disable_checksum=True)
    x = arr.reshape(-1).astype(np.float64)
    err = np.abs(out.astype(np.float64) - x)
    assert (err <= 0.5 * np.sqrt(4.0 + x)).all()