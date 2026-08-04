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


# The bounds are cut and checked in C. What is left here is that a node placed
# by hand reaches those kernels with the parameters the recipe asked for.
def _lossy(node, arr):
    out = roundtrip(node, arr, disable_checksum=True)
    return out.astype(np.float64), arr.reshape(-1).astype(np.float64)


def test_quant_linear_holds_an_absolute_bound():
    arr = make_tile((16, 16), np.float32, "gradient")
    out, src = _lossy(
        geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.float32), arr)
    assert np.abs(out - src).max() <= 0.5


def test_quant_log_holds_a_relative_bound():
    # six decades, a relative bound on a narrow tile is an absolute one
    arr = np.logspace(-3, 3, 256).astype(np.float32).reshape(16, 16)
    out, src = _lossy(geozl.lossy.QuantLog("LOG:MAX_ERROR=1%", np.float32), arr)
    assert (np.abs(out - src) / np.abs(src)).max() <= 0.01


def test_quant_sqrt_holds_the_curve_in_its_recipe():
    arr = make_tile((16, 16), np.uint16, "gradient")
    out, src = _lossy(
        geozl.lossy.QuantSqrt("SQRT:MAX_ERROR=0.5N,A=100,B=1", np.uint16), arr)
    assert (np.abs(out - src) / (0.5 * np.sqrt(100.0 + src))).max() <= 1.0


def test_a_quantizer_refuses_a_recipe_from_another_family():
    with pytest.raises(ValueError, match="quant_log takes"):
        geozl.lossy.QuantLog("LINEAR:MAX_ERROR=1", np.float32)


def test_a_quantizer_refuses_to_write_under_a_content_checksum():
    # the round trip is not bit exact, so the hash would fail on a good frame
    arr = make_tile((16, 16), np.float32, "gradient")
    with pytest.raises(Exception, match="ContentChecksum"):
        compress(geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.float32),
                 arr)


def test_a_quantizer_refuses_a_dtype_it_has_no_kernel_for():
    with pytest.raises(ValueError, match="has no kernel"):
        geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.complex64)


def test_a_quantizer_refuses_a_dtype_that_is_not_the_stream_width():
    # the dtype is given at construction, the stream carries a width
    arr = make_tile((16, 16), np.float32, "gradient")
    with pytest.raises(Exception, match="does not match"):
        compress(geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.float64),
                 arr, disable_checksum=True)
