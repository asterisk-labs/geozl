import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

# The curve the synthetic rasters are drawn from, so a fit that works recovers
# these. b is the robust half; a is an intercept extrapolated past where the data
# reaches and moves several fold, which is why nothing here pins it tightly.
_A, _B = 100.0, 1.0


def _scene(n, lo, hi, seed=0):
    """A raster whose local variance really is _A + _B*x, sorted so the
    intensity varies smoothly and the high pass reads noise rather than edges."""
    rng = np.random.default_rng(seed)
    mu = np.sort(rng.uniform(lo, hi, n * n)).reshape(n, n)
    return (mu + rng.normal(0.0, np.sqrt(_A + _B * mu))).astype(np.float32)


def test_fit_recovers_the_curve_from_one_raster():
    noise = geozl.lossy.fit_noise(_scene(512, 50, 4000))
    assert 0.8 <= noise.b <= 1.3
    assert noise.bins >= 4
    assert noise.blocks > 0


def test_a_stack_fits_as_one_curve():
    stack = np.stack([_scene(256, 50, 4000, seed=i) for i in range(8)])
    assert stack.shape == (8, 256, 256)
    noise = geozl.lossy.fit_noise(stack)
    assert 0.8 <= noise.b <= 1.3
    # Every raster reached the pool, so this holds more blocks than one does.
    assert noise.blocks > geozl.lossy.fit_noise(stack[0]).blocks


def test_pooling_beats_the_rasters_it_pools():
    # The case the accumulator exists for. A tile covering only the bright end
    # has no leverage to separate a from b, so it returns a plausible curve that
    # is not the sensor's, and only colin says so. Pooling it with the dark end
    # recovers both.
    dark, bright = _scene(256, 50, 300, seed=1), _scene(256, 3000, 4000, seed=2)
    alone = geozl.lossy.fit_noise(bright)
    pooled = geozl.lossy.fit_noise([dark, bright])
    assert alone.colin > 5.0
    assert pooled.colin < alone.colin
    assert abs(pooled.b - _B) < abs(alone.b - _B)


def test_a_sequence_and_a_stack_agree():
    rasters = [_scene(128, 50, 4000, seed=i) for i in range(4)]
    a = geozl.lossy.fit_noise(rasters)
    b = geozl.lossy.fit_noise(np.stack(rasters))
    assert (a.a, a.b) == (b.a, b.b)


def test_recipe_round_trips_the_curve():
    noise = geozl.lossy.fit_noise(_scene(256, 50, 4000))
    recipe = noise.recipe(0.5)
    assert recipe.startswith("SQRT:MAX_ERROR=0.5N,A=")
    # Full precision, so the recipe resolves to the grid the fit produced rather
    # than to a nearby one.
    a = float(recipe.split("A=")[1].split(",")[0])
    b = float(recipe.split("B=")[1])
    assert (a, b) == (noise.a, noise.b)
    assert noise.recipe(1, store="VALUES").endswith(",STORE=VALUES")


def test_recipe_refuses_what_the_codec_would():
    noise = geozl.lossy.fit_noise(_scene(256, 50, 4000))
    with pytest.raises(ValueError):
        noise.recipe(0)
    with pytest.raises(ValueError):
        noise.recipe(0.5, store="MAYBE")


def test_a_flat_raster_has_no_curve():
    with pytest.raises(RuntimeError, match="dynamic range"):
        geozl.lossy.fit_noise(np.zeros((64, 64), np.float32))


def test_a_raster_too_small_to_hold_a_block_is_refused():
    with pytest.raises(ValueError, match="too small"):
        geozl.lossy.fit_noise(np.zeros((4, 4), np.float32))


def test_rasters_of_different_shapes_pool():
    small, large = _scene(64, 50, 4000, seed=3), _scene(256, 50, 4000, seed=4)
    assert geozl.lossy.fit_noise([small, large]).blocks > 0


def test_mixed_dtypes_are_refused():
    a = _scene(64, 50, 4000)
    with pytest.raises(ValueError, match="share a dtype"):
        geozl.lossy.fit_noise([a, a.astype(np.float64)])


def test_a_dtype_with_no_kernel_is_refused():
    with pytest.raises(ValueError, match="does not support"):
        geozl.lossy.fit_noise(np.zeros((64, 64), np.bool_))


@pytest.mark.parametrize("ndim", [1, 4])
def test_only_2d_and_3d_arrays_are_taken(ndim):
    with pytest.raises(ValueError, match="dimensions"):
        geozl.lossy.fit_noise(np.zeros((16,) * ndim, np.float32))


def test_an_integer_raster_fits():
    arr = _scene(256, 200, 4000).astype(np.uint16)
    assert geozl.lossy.fit_noise(arr).b > 0.0
