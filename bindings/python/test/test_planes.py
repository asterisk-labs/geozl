"""Stacked planes. A predictor that reads the row above restarts at each plane
boundary, so a (B, Y, X) cube is predicted band by band while everything after
it in the graph still sees one stream."""

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

# graph, compress and profile live in libgeozl, which FULL=OFF does not build.
try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

# delta_w only reads to its left, so planes cannot change what it writes
LOOKS_UP = ["planar", "delta_n", "average", "med", "wp_static"]
ALL_PREDICTORS = LOOKS_UP + ["delta_w"]


def cube(shape=(4, 32, 32), dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3000, shape).astype(dtype)


def roundtrip(arr, method, **kw):
    g = geozl.graph(arr, method, **kw)
    frame = geozl.compress(arr, graph=g)
    back = geozl.decompress(frame).view(arr.dtype).reshape(arr.shape)
    return g, frame, back


@pytest.mark.parametrize("pred", ALL_PREDICTORS)
def test_a_cube_round_trips(pred):
    a = cube()
    _g, _f, back = roundtrip(a, f"{pred}>zigzag>zstd")
    assert np.array_equal(back, a)


def test_planes_comes_from_the_first_axis():
    assert geozl.graph(cube((5, 16, 16)), "planar>zigzag>zstd").planes == 5
    assert geozl.graph(cube((16, 16)), "planar>zigzag>zstd").planes == 1
    assert geozl.graph(cube((2, 3, 8, 8)), "planar>zigzag>zstd").planes == 2


def test_width_still_comes_from_the_last_axis():
    assert geozl.graph(cube((4, 12, 20)), "planar>zigzag>zstd").width == 20


@pytest.mark.parametrize("pred", LOOKS_UP)
def test_planes_changes_the_frame(pred):
    """If the two agree the parameter never reached the kernel."""
    a = cube()
    m = f"{pred}>zigzag>zstd"
    assert (geozl.compress(a, graph=geozl.graph(a, m))
            != geozl.compress(a, graph=geozl.graph(a, m, planes=1)))


def test_delta_w_ignores_planes():
    a = cube()
    m = "delta_w>zigzag>zstd"
    assert (geozl.compress(a, graph=geozl.graph(a, m))
            == geozl.compress(a, graph=geozl.graph(a, m, planes=1)))


@pytest.mark.parametrize("pred", ALL_PREDICTORS)
def test_one_plane_is_the_old_wire(pred):
    """A 2D raster must produce the bytes it produced before planes existed,
    which is what keeps every frame already written readable."""
    a = cube((32, 32))
    g = geozl.graph(a, f"{pred}>zigzag>zstd")
    assert g.planes == 1
    explicit = geozl.graph(a, f"{pred}>zigzag>zstd", planes=1)
    assert (geozl.compress(a, graph=g)
            == geozl.compress(a, graph=explicit))


@pytest.mark.parametrize("planes", [1, 2, 3, 4, 6, 12])
def test_every_divisor_round_trips(planes):
    a = cube((12, 24, 24))
    _g, _f, back = roundtrip(a, "planar>zigzag>zstd", planes=planes)
    assert np.array_equal(back, a)


@pytest.mark.parametrize("planes", [0, -1, 5, 7, 100])
def test_a_count_that_does_not_split_the_stream_is_refused(planes):
    with pytest.raises(ValueError):
        geozl.graph(cube((4, 32, 32)), "planar>zigzag>zstd", planes=planes)


@pytest.mark.parametrize("shape", [(1, 8, 8), (2, 1, 16), (2, 16, 1),
                                   (3, 1, 1), (8, 17, 33), (64, 4, 4)])
def test_awkward_cube_shapes(shape):
    a = cube(shape)
    _g, _f, back = roundtrip(a, "planar>zigzag>zstd")
    assert np.array_equal(back, a)


@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16",
                                   "uint32", "int32", "uint64", "int64"])
def test_every_integer_width(dtype):
    a = cube((3, 16, 16), dtype=dtype)
    _g, _f, back = roundtrip(a, "planar>zigzag>zstd")
    assert np.array_equal(back, a)


def test_float_cube_with_nodata_in_one_band_only():
    rng = np.random.default_rng(4)
    a = rng.normal(1000, 200, (3, 16, 16)).astype(np.float32)
    a[1, 4:8, 4:8] = np.nan
    _g, _f, back = roundtrip(a, "planar>zigzag>zstd")
    assert np.array_equal(back, a, equal_nan=True)


def test_a_lossy_cube_keeps_its_shape():
    a = cube((3, 16, 16))
    _g, _f, back = roundtrip(a, "planar>zigzag>zstd",
                             error="LINEAR:MAX_ERROR=4")
    assert back.shape == a.shape
    assert np.abs(back.astype(np.int64) - a.astype(np.int64)).max() <= 4


def test_profile_accepts_a_cube():
    rows = geozl.profile(cube((4, 32, 32)))
    assert rows and "graph" in rows[0]


def test_a_corrupt_plane_count_is_refused():
    """The count rides in the codec header, so a bad one is corruption rather
    than a caller mistake."""
    a = cube((4, 16, 16))
    frame = bytearray(geozl.compress(a, graph=geozl.graph(a, "planar>zigzag>zstd")))
    hit = 0
    for i in range(len(frame)):
        b = bytearray(frame)
        b[i] ^= 0xFF
        try:
            out = geozl.decompress(bytes(b)).view(a.dtype)
        except Exception:
            continue
        if out.size == a.size and np.array_equal(out.reshape(a.shape), a):
            continue
        hit += 1
    assert hit == 0, f"{hit} single byte flips decoded to different data"
