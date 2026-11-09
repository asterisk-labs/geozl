"""The public surface the API page describes, past what the signatures show."""

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d, _ffi  # noqa: E402  after importorskip, on purpose

try:
    _ffi._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

_DTYPES = ["uint8", "int8", "uint16", "int16", "float16", "uint32", "int32",
           "float32", "uint64", "int64", "float64", "complex64"]


def _noise(dtype, n, seed=0):
    """Uniformly random bytes, so no codec finds anything to remove."""
    raw = np.random.default_rng(seed).integers(0, 256, n * np.dtype(dtype).itemsize,
                                               dtype=np.uint8)
    return raw.view(dtype)


def _bound(nbytes, coeffs=0):
    return int(_ffi._load_lib_full().geozl_2d_compress_bound_c(nbytes, coeffs))


# dstCapacity_tooSmall and internalBuffer_tooSmall, the failures a short buffer
# causes. A recipe may fail on noise for other reasons, such as entropy on
# samples wider than 2 bytes.
_SHORT = ("(ZL error code 5)", "(ZL error code 71)")


def _frame(tile, g, **kwargs):
    try:
        return geozl.compress(tile, graph=g, **kwargs)
    except RuntimeError as exc:
        assert not str(exc).endswith(_SHORT), str(exc)
        return None


def test_graph_and_profile_results_are_returned_types():
    tile = np.arange(64, dtype=np.uint16).reshape(8, 8)
    g = geozl.graph(tile, "planar>zigzag>entropy")
    rows = geozl.profile(tile, reps=1)
    assert isinstance(g, geozl.Graph)
    assert isinstance(rows, geozl.ProfileResults)
    assert rows.width == 8 and rows.reps == 1
    with pytest.raises(TypeError, match="returned by geozl.graph"):
        geozl.Graph()
    with pytest.raises(TypeError, match="returned by geozl.graph"):
        geozl.Graph(None, method="id>zstd", width=8, planes=1,
                    dtype=tile.dtype, itemsize=2, error=None, nodata=None)
    with pytest.raises(TypeError, match="returned by geozl.profile"):
        geozl.ProfileResults()


def test_a_sqrt_node_without_its_curve_is_refused_when_built():
    pytest.importorskip("openzl")
    with pytest.raises(ValueError, match="no A and B"):
        geozl.lossy.QuantSqrt("SQRT:MAX_ERROR=1N", np.uint16)
    with pytest.raises(ValueError, match="A and B"):
        geozl.lossy.QuantSqrt("SQRT:MAX_ERROR=1N,A=4", np.uint16)
    geozl.lossy.QuantSqrt("SQRT:MAX_ERROR=1N,A=4,B=1", np.uint16)


def test_an_empty_frame_and_an_unreadable_one_are_told_apart():
    lib = _ffi._load_lib_full()
    empty = geozl.compress(np.zeros(0, np.uint16),
                           graph=geozl.graph(np.zeros(0, np.uint16), "id>zstd", width=1))
    size = _ffi.ffi.new("size_t*", 7)
    buf = np.frombuffer(empty, np.uint8)
    assert lib.geozl_2d_frame_dsize_c(_ffi._ptr(buf), buf.size, size) == 0
    assert size[0] == 0
    assert geozl.decompress(empty).size == 0

    junk = np.frombuffer(b"\x01\x02\x03\x04" * 4, np.uint8)
    size[0] = 7
    assert lib.geozl_2d_frame_dsize_c(_ffi._ptr(junk), junk.size, size) != 0
    assert size[0] == 0
    with pytest.raises(RuntimeError, match="unreadable frame"):
        geozl.decompress(junk.tobytes())


def test_the_bound_refuses_sizes_that_overflow():
    assert _bound(2**64 - 1) == 0
    assert _bound(0, 2**64 - 1) == 0
    assert _bound(0) > 0
    assert _bound(1000, 10) == _bound(1000) + 14


@pytest.mark.parametrize("dtype", _DTYPES)
def test_no_recipe_writes_past_the_bound_on_noise(dtype):
    """Noise at every dtype and every recipe the grid holds, the worst case for
    each terminal, plus the sizes where fixed overhead dominates."""
    elt = np.dtype(dtype).itemsize
    written = 0
    for n in (0, 1, 3, 255, 4096, 70_000):
        tile = _noise(dtype, n, seed=n)
        for name in _2d._grid_names(None, elt):
            frame = _frame(tile, geozl.graph(tile, name, width=max(n, 1)))
            if frame is not None:
                assert len(frame) <= _bound(tile.nbytes), (name, n)
                written += 1
    assert written > 100


@pytest.mark.parametrize("dtype", ["uint8", "int16", "uint16", "float32", "float64"])
@pytest.mark.parametrize("error", [2, "1%"])
def test_lossy_and_nodata_frames_stay_inside_the_bound(dtype, error):
    rng = np.random.default_rng(3)
    if np.dtype(dtype).kind == "f":
        tile = rng.uniform(0, 1000, (64, 64)).astype(dtype)
    else:
        tile = _noise(dtype, 64 * 64, seed=3).reshape(64, 64)
    nodata = tile.flat[0]
    written = 0
    for name in ("planar>zigzag>entropy", "id>zstd", "planar>zigzag>pfor",
                 "id>lanes", "delta_1d>sparse"):
        frame = _frame(tile, geozl.graph(tile, name, error=error, nodata=nodata))
        if frame is not None:
            assert len(frame) <= _bound(tile.nbytes), name
            written += 1
    assert written >= 3


def test_coefficients_stay_inside_the_bound():
    tile = _noise("uint16", 4096, seed=5)
    g = geozl.graph(tile, "id>zstd", width=64)
    coeffs = [list(range(-40, 40))] * 30
    blob = len(geozl._coeffs._pack_coeffs(coeffs))
    frame = geozl.compress(tile, graph=g, coeffs=coeffs)
    assert len(frame) <= _bound(tile.nbytes, blob)
    assert geozl.coeffs(frame) is not None
