import struct

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

# The signaled kernel path, a fixed kernel and shift, is no longer reachable from
# Python, WpStaticInt takes only the width. It is exercised at the C level with
# wp_static_encode and wp_static_decode directly.


def _frame(node, arr):
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = node(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def _roundtrip(node, arr):
    frame = _frame(node, arr)
    d = zl.DCtx()
    geozl.lossless.register_decoders(d)
    geozl.lossy.register_decoders(d)
    out = d.decompress(frame)[0].content.as_nparray()
    return out.view(arr.dtype)


def _make_tile(shape, dtype, pattern):
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


def _anisotropic(h, w, seed):
    """A field stretched along one diagonal, so the plane predictor is not
    optimal and a trained kernel can beat it."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    u, v = kx + ky, kx - ky
    spec = np.fft.fft2(rng.normal(size=(h, w)))
    field = np.fft.ifft2(spec / (1 + (u / 0.02) ** 2 + (v / 0.16) ** 2) ** 1.5).real
    field = (field - field.mean()) / (field.std() + 1e-9)
    return np.clip(np.round(28000 + 6000 * field), 0, 65535).astype(np.uint16)


_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64,
           np.int8, np.int16, np.int32, np.int64,
           np.float16, np.float32, np.float64]
_SHAPES = [(16, 16), (7, 13), (1, 32), (32, 1), (4, 4)]
_PATTERNS = ["random", "gradient", "constant"]


# The autonomous codec round trips bit exact on every width and shape, whatever
# kernel it fits, exactly like the planar default it falls back to.
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_round_trip(dtype, shape, pattern):
    arr = _make_tile(shape, dtype, pattern)
    out = _roundtrip(geozl.lossless.WpStaticInt(shape[1]), arr)
    assert np.array_equal(out, arr.reshape(-1))


# On a directional tile the trained kernel beats the plane predictor by more than
# the nine extra header bytes it costs, so the C trainer really does fit.
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_training_beats_planar(seed):
    arr = _anisotropic(256, 256, seed)
    wp = len(_frame(geozl.lossless.WpStaticInt(256), arr))
    planar = len(_frame(geozl.lossless.PlanarInt(256), arr))
    assert wp < planar


# On plane friendly data the trainer falls back, so wp_static never costs more
# than its header overhead over planar, it does not settle on a worse kernel.
def test_fallback_not_worse():
    arr = _make_tile((64, 64), np.uint16, "gradient")
    wp = len(_frame(geozl.lossless.WpStaticInt(64), arr))
    planar = len(_frame(geozl.lossless.PlanarInt(64), arr))
    assert wp <= planar + 16


# The layout of the wp_static header, matched to header_wp_static in C, only so
# the test can read back the kernel the C trainer chose.
_HEADER = "<IB4h"


def _trained_kernel(arr):
    """Compress arr with wp_static and read the kernel the C trainer wrote into
    the codec header. Returns (coeffs, shift)."""
    seen = {}

    class Spy(geozl.lossless.WpStaticDecoder):
        def decode(self, state):
            width, shift, *coeffs = struct.unpack(_HEADER, state.codec_header)
            seen["kernel"] = (tuple(coeffs), shift)
            super().decode(state)

    frame = _frame(geozl.lossless.WpStaticInt(arr.shape[1]), arr)
    d = zl.DCtx()
    d.register_custom_decoder(Spy())
    d.decompress(frame)
    return seen["kernel"]


# The kernel the C trainer picks must reach an entropy at least as good as the
# numpy oracle on the same tile. The coefficients may differ, ridge versus SVD,
# what has to match is the entropy they achieve, measured the same way for both.
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trained_kernel_matches_oracle(seed):
    oracle = pytest.importorskip("oracle_wp_static")
    arr = _anisotropic(256, 256, seed)

    c_coeffs, c_shift = _trained_kernel(arr)
    assert (c_coeffs, c_shift) != ((1, -1, 0, 0), 0)      # it really trained

    o_coeffs, o_shift = oracle.fit_wp_static(arr)
    c_bits = oracle._plane_bits(oracle._residual(arr, c_coeffs, c_shift))
    o_bits = oracle._plane_bits(oracle._residual(arr, o_coeffs, o_shift))

    assert c_bits <= o_bits + 0.05                        # no worse than the oracle