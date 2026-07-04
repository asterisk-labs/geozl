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
    rng = np.random.default_rng(abs(hash((shape, str(dtype)))) % (2**32))
    return (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)


def encode(z):
    width = z.shape[1]
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = zl.nodes.Zigzag()(c, g)
    g = geozl.lossless.DeltaWInt(width)(c, g)        # a geozl predictor per component
    g = geozl.lossless.complex_split(z.dtype)(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    comp = geozl.lossless.component_dtype(z.dtype)
    flat = np.ascontiguousarray(z).reshape(-1).view(comp)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


_DTYPES = [np.complex64, np.complex128]
_SHAPES = [(16, 16), (7, 13), (1, 32), (32, 1)]


# pass the dtype, so a small tile that stores raw (its frame overhead beats the
# payload, dropping the codec) still round trips. When the codec rides in the
# frame the hint is ignored.
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", ["gradient", "constant"])
def test_complex_bit_exact(dtype, shape, pattern):
    z = make_complex(shape, dtype, pattern)
    back = geozl.lossless.decompress_complex(encode(z), dtype=dtype)
    assert back.dtype == np.dtype(dtype)
    assert np.array_equal(back, np.ascontiguousarray(z).reshape(-1))


# a tile large enough to compress keeps complex_split in the frame, so the dtype
# comes back on its own with no hint
def test_complex_dtype_rides_in_frame():
    z = make_complex((64, 64), np.complex64, "gradient")
    back = geozl.lossless.decompress_complex(encode(z))
    assert back.dtype == np.complex64
    assert np.array_equal(back, np.ascontiguousarray(z).reshape(-1))


# a frame without complex_split, the state OpenZL leaves after storing the tile
# raw on incompressible data, carries no complex dtype, so the caller passes it
def test_complex_dtype_hint():
    z = make_complex((16, 16), np.complex128, "gradient")
    comp = geozl.lossless.component_dtype(z.dtype)
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(z).reshape(-1).view(comp)
    frame = bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))

    with pytest.raises(ValueError):
        geozl.lossless.decompress_complex(frame)
    back = geozl.lossless.decompress_complex(frame, dtype=np.complex128)
    assert np.array_equal(back, np.ascontiguousarray(z).reshape(-1))