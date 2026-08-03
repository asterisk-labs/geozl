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


# The lossy codecs have no Python encoder classes yet. Their bounds are covered
# in C, in test_quant_linear.c, test_quant_log.c and test_quant_sqrt.c, and
# through geozl.compress in test_2d.py.
