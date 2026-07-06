import zlib

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")


def make_complex(shape, dtype, pattern):
    n = shape[0] * shape[1]
    if pattern == "constant":
        return np.full(shape, 3 + 4j, dtype=dtype)
    if pattern == "gradient":
        re = np.linspace(0.0, 100.0, n)
        im = np.linspace(50.0, 150.0, n)
        return (re + 1j * im).reshape(shape).astype(dtype)
    rng = np.random.default_rng(zlib.crc32(f"{shape}-{dtype}".encode()))
    return (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)


def encode_complex(z):
    # split real and imaginary before predicting, so the predictor never crosses
    # the component boundary. deinterleave knows nothing of complex, the caller
    # views the tile as its component float first.
    w = z.shape[1]
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = zl.nodes.Zigzag()(c, g)
    g = geozl.lossless.DeltaW(w)(c, g)
    g = geozl.lossless.Deinterleave()(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    comp = geozl.lossless.component_dtype(z.dtype)
    flat = np.ascontiguousarray(z).reshape(-1).view(comp)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def decode_complex(frame, dtype):
    d = zl.DCtx()
    geozl.register_decoders(d)
    out = d.decompress(frame)[0].content.as_nparray()
    # the decoded bytes are [re, im, re, im], which is the complex memory layout
    return out.view(dtype)


_CPLX = [np.complex64, np.complex128]
_SHAPES = [(16, 16), (7, 13), (1, 32), (32, 1)]


@pytest.mark.parametrize("dtype", _CPLX, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", ["gradient", "constant", "random"])
def test_complex_bit_exact(dtype, shape, pattern):
    z = make_complex(shape, dtype, pattern)
    back = decode_complex(encode_complex(z), dtype)
    assert back.dtype == np.dtype(dtype)
    assert np.array_equal(back, np.ascontiguousarray(z).reshape(-1))


def _deinterleave_roundtrip(arr):
    # deinterleave straight into compress, no predictor, so only the split and the
    # rejoin are under test
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = geozl.lossless.Deinterleave()(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    frame = bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))
    d = zl.DCtx()
    geozl.register_decoders(d)
    out = d.decompress(frame)[0].content.as_nparray()
    return out.view(arr.dtype)


# an odd count leaves the even lane one element longer than the odd lane, the tail
# case a complex tile never reaches since components come in pairs
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("n", [2, 3, 8, 9, 64, 65])
def test_deinterleave_numeric(dtype, n):
    rng = np.random.default_rng(zlib.crc32(f"{dtype}-{n}".encode()))
    info = np.iinfo(dtype)
    arr = rng.integers(info.min, info.max, size=n, endpoint=True, dtype=dtype)
    out = _deinterleave_roundtrip(arr)
    assert np.array_equal(out, arr)
