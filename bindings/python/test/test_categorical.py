"""The categorical terminal. One pass over the stream picks constant, field_lz
or entropy from the share the dominant symbol holds, and OpenZL records which
one it took, so there is no header here and no decoder to register.
"""
import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

CAT = "id>categorical"
W = 256


def _classes(dominance, classes=8, dtype=np.uint8, edge=W):
    """A field where one class takes `dominance` of it and the rest split the
    remainder. Not blobs, so there is little run structure to find."""
    rng = np.random.default_rng(0)
    out = rng.integers(1, classes, (edge, edge))
    out[rng.random((edge, edge)) < dominance] = 0
    return out.astype(dtype)


def _size(arr, method=CAT):
    return len(geozl.compress(arr, graph=geozl.graph(arr, method)))


def _roundtrip(arr, method=CAT):
    frame = geozl.compress(arr, graph=geozl.graph(arr, method))
    return geozl.decompress(frame).view(arr.dtype).reshape(arr.shape)


@pytest.mark.parametrize("dominance", [1.0, 0.99, 0.5, 0.0])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int8, np.int16],
                         ids=lambda d: np.dtype(d).name)
def test_every_branch_round_trips(dominance, dtype):
    arr = _classes(dominance, dtype=dtype)
    assert np.array_equal(_roundtrip(arr), arr)


def test_a_single_class_takes_the_constant_arm():
    """84 per cent of the patches this exists for hold one value, and that arm
    carries most of the saving. It has to stay flat as the raster grows, which
    is what tells it apart from an entropy backend that merely does well."""
    small = np.full((64, 64), 3, np.uint8)
    large = np.full((1024, 1024), 3, np.uint8)
    assert _size(large) < 64
    assert _size(large) - _size(small) < 8
    assert _size(large) < _size(large, "id>entropy") / 4


def test_the_constant_arm_survives_a_predictor_in_front_of_it():
    """With a predictor the branch sees the residual, not the classes, and a
    constant raster has a constant residual too."""
    arr = np.full((256, 256), 7, np.uint8)
    assert _size(arr, "delta_n>zigzag>categorical") < 64
    assert np.array_equal(_roundtrip(arr, "delta_n>zigzag>categorical"), arr)


def test_a_skewed_field_does_not_lose_to_either_fixed_terminal_by_much():
    """The dispatch cannot beat whichever fixed terminal happens to win, it can
    only avoid being far from it without being told which raster it has."""
    for dominance in (1.0, 0.99, 0.5):
        arr = _classes(dominance)
        best = min(_size(arr, "id>entropy"), _size(arr, "id>field_lz"))
        assert _size(arr) <= best * 1.8, dominance


def test_the_smallest_raster_still_takes_a_branch():
    """One sample is constant by definition."""
    for n in (1, 2, 3):
        arr = np.full(n, 5, np.uint8)
        frame = geozl.compress(arr, graph=geozl.graph(arr, CAT, width=1))
        assert np.array_equal(geozl.decompress(frame), arr)


def test_it_is_in_the_grid_where_a_class_raster_lives():
    rows = geozl.profile(_classes(0.9, edge=64), prior=None, reps=1)
    assert any(r["graph"] == CAT for r in rows)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_it_does_not_apply_past_two_bytes(dtype):
    """Its entropy arm tops out at 2 bytes, and the other two take any width, so
    without a cap the same method would compress a raster whose dominant share
    is high and refuse the same raster once it is not."""
    arr = _classes(0.5, dtype=dtype, edge=32)
    with pytest.raises(RuntimeError, match="categorical needs 1 or 2"):
        geozl.graph(arr, CAT)
