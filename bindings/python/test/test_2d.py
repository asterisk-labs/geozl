import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

# The 2d entries live in libgeozl, which a FULL=OFF build does not produce.
try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

# Wins on smooth integer rasters, and what the docs lead with.
GRAPH = "planar>zigzag>transpose>entropy"

# transpose needs 2 to 8 bytes per element, so 1-byte tiles need this instead.
GRAPH_1B = "planar>zigzag>entropy"


def _tile(shape=(32, 32), dtype=np.int16):
    """A smooth ramp with a little noise: compressible, but not trivially so."""
    rng = np.random.default_rng(0)
    y, x = np.indices(shape)
    return ((x * 3 + y * 5) % 400 + rng.integers(0, 4, shape)).astype(dtype)


def _roundtrip(arr, **kw):
    frame = geozl.compress(arr, **kw)
    return geozl.decompress(frame, dtype=arr.dtype.name, width=arr.shape[1])


def test_rejects_non_native_byte_order():
    with pytest.raises(ValueError, match="native byte order"):
        geozl.compress(_tile().astype(">i2"), method=GRAPH)


def test_rejects_1d_without_width():
    with pytest.raises(ValueError, match="width"):
        geozl.compress(np.arange(64, dtype=np.int16), method=GRAPH)


def test_1d_with_width_encodes_like_the_2d_tile():
    arr = _tile()
    flat = geozl.compress(arr.reshape(-1), method=GRAPH, width=arr.shape[1])
    assert flat == geozl.compress(arr, method=GRAPH)


def test_rejects_an_element_width_openzl_cannot_stream():
    # complex128 is 16 bytes, a numeric stream is 1, 2, 4 or 8
    with pytest.raises(ValueError, match="bytes per element"):
        geozl.compress(np.zeros((4, 4), np.complex128), method=GRAPH)


def test_method_is_required():
    with pytest.raises(TypeError):
        geozl.compress(_tile())


@pytest.mark.parametrize("bad", [None, "", 123, b"planar>zigzag>entropy"])
def test_method_must_be_a_recipe_string(bad):
    with pytest.raises(ValueError, match="must be a recipe name"):
        geozl.compress(_tile(), method=bad)


def test_unknown_recipe_is_rejected_by_the_library():
    with pytest.raises(RuntimeError, match="unknown method"):
        geozl.compress(_tile(), method="nope>zigzag>entropy")


def test_recipe_that_does_not_fit_the_element_width_fails():
    # the point of dropping the search: no silent fallback to something else
    with pytest.raises(RuntimeError, match="1-byte elements"):
        geozl.compress(_tile(dtype=np.uint8), method=GRAPH)


def test_compress_is_deterministic():
    arr = _tile()
    assert geozl.compress(arr, method=GRAPH) == geozl.compress(arr, method=GRAPH)


def test_compress_actually_compresses():
    arr = _tile()
    assert len(geozl.compress(arr, method=GRAPH)) < arr.nbytes


def test_no_error_is_lossless():
    arr = _tile()
    assert np.array_equal(_roundtrip(arr, method=GRAPH, error=None), arr)


# A number used to mean an absolute bound, and 0 or less meant lossless. Both
# readings are gone: the bound picks the curve, so it has to say which one.
@pytest.mark.parametrize("error", [0, -1, 1, 2.0])
def test_a_bare_number_is_not_an_error_recipe(error):
    with pytest.raises(ValueError, match="recipe string"):
        geozl.compress(_tile(), method=GRAPH, error=error)


def test_compress_rejects_a_dtype_quant_has_no_kernel_for():
    with pytest.raises(ValueError, match="quant does not support"):
        geozl.compress(np.zeros((4, 4), np.bool_), method=GRAPH_1B, error="abs:2")


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint32, np.float64])
def test_round_trip_is_bit_exact(dtype):
    arr = _tile(dtype=np.int32).astype(dtype)
    graph = GRAPH_1B if arr.dtype.itemsize == 1 else GRAPH
    out = _roundtrip(arr, method=graph)
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


def test_without_dtype_you_get_an_unsigned_view():
    # the frame keeps the bytes but not the sign, and no width leaves it flat
    arr = _tile()
    back = geozl.decompress(geozl.compress(arr, method=GRAPH))
    assert back.ndim == 1
    assert back.nbytes == arr.nbytes


def test_without_width_the_array_stays_flat():
    arr = _tile()
    back = geozl.decompress(geozl.compress(arr, method=GRAPH), dtype="int16")
    assert back.shape == (arr.size,)


def test_unreadable_frame_is_reported():
    with pytest.raises(RuntimeError, match="unreadable frame"):
        geozl.decompress(b"not a frame at all")


def test_corrupt_payload_is_reported():
    frame = bytearray(geozl.compress(_tile(), method=GRAPH))
    frame[-8] ^= 0xFF  # past the header, so the size still reads
    with pytest.raises(RuntimeError, match="decompress failed"):
        geozl.decompress(bytes(frame), dtype="int16")


@pytest.mark.parametrize("recipe,bound", [("abs:1", 1), ("abs:2.0", 2.0),
                                          ("abs:7", 7)])
def test_error_bound_holds(recipe, bound):
    arr = _tile()
    out = _roundtrip(arr, method=GRAPH, error=recipe)
    assert np.abs(out.astype(np.int64) - arr.astype(np.int64)).max() <= bound


def test_lossy_beats_lossless_on_size():
    arr = _tile()
    assert len(geozl.compress(arr, method=GRAPH, error="abs:8")) < \
           len(geozl.compress(arr, method=GRAPH))


def test_profile_returns_one_row_per_graph_sorted_by_ratio():
    rows = geozl.profile(_tile(), reps=1)
    keys = {"graph", "ratio", "encode_mbps", "decode_mbps", "shannon_pct"}
    assert rows and all(keys <= set(r) for r in rows)
    ratios = [r["ratio"] for r in rows]
    assert ratios == sorted(ratios, reverse=True)
    assert len({r["graph"] for r in rows}) == len(rows)


def test_profile_names_graphs_compress_accepts():
    arr = _tile()
    best = geozl.profile(arr, reps=1)[0]["graph"]
    assert np.array_equal(_roundtrip(arr, method=best), arr)


def test_profile_ratio_matches_the_frame_compress_produces():
    arr = _tile()
    row = geozl.profile(arr, reps=1)[0]
    frame = geozl.compress(arr, method=row["graph"])
    assert len(frame) == pytest.approx(arr.nbytes / row["ratio"], rel=0.01)


def test_unbiased_prior_sweeps_more_graphs_than_a_named_one():
    arr = _tile()
    assert len(geozl.profile(arr, method=None, reps=1)) > \
           len(geozl.profile(arr, method="planar", reps=1))


def test_unbiased_prior_is_the_whole_grid():
    # 8 predictors x 6 terminals, all buildable at 2 bytes per element
    assert len(geozl.profile(_tile(), method=None, reps=1)) == 48


def test_transpose_terminals_drop_out_on_1_byte_elements():
    rows = geozl.profile(_tile(dtype=np.uint8), method=None, reps=1)
    assert rows
    assert not any("transpose" in r["graph"] or "store_lo" in r["graph"]
                   for r in rows)


def test_profile_rejects_an_unknown_prior():
    with pytest.raises(ValueError, match="not one of"):
        geozl.profile(_tile(), method="not_a_predictor")


def test_profile_lossy_beats_profile_lossless():
    arr = _tile()
    assert geozl.profile(arr, reps=1, error="abs:8")[0]["ratio"] > \
           geozl.profile(arr, reps=1)[0]["ratio"]


def test_profile_rejects_a_dtype_quant_has_no_kernel_for():
    with pytest.raises(ValueError, match="quant does not support"):
        geozl.profile(np.zeros((4, 4), np.bool_), error="abs:2")


def test_profile_skips_a_graph_that_fails_on_this_tile(monkeypatch):
    # a 1-byte tile cannot run transpose, so a grid offering it must skip that
    # row rather than abort the whole profile
    arr = _tile(dtype=np.uint8)
    grid = _2d._grid_names
    monkeypatch.setattr(_2d, "_grid_names", lambda m, e: [GRAPH] + grid(m, e))
    rows = _2d.profile(arr, reps=1)
    assert rows and all(r["graph"] != GRAPH for r in rows)


def test_shannon_pct_passes_100_when_structure_is_found():
    # a smooth ramp gives a spatial predictor plenty, so the frame has to beat
    # the order-0 entropy of the raw bytes
    assert geozl.profile(_tile(), reps=1)[0]["shannon_pct"] > 100.0


def test_order0_bits_of_a_constant_tile_is_zero():
    assert _2d._order0_bits(np.zeros((8, 8), np.uint8)) == 0.0


# nodata. The behaviour of the codec itself lives in test_nodata.py, what
# belongs here is the argument surface of _2d and, above all, that a tile with
# nothing missing is untouched by any of it.

def _holed(dtype=np.float32, hole=np.nan):
    """Bigger than _tile on purpose. profile times with the content checksum
    off, so its frame runs four bytes under the one compress writes, and a
    small tile would spend the whole 1% tolerance on those four bytes."""
    arr = _tile(shape=(64, 64), dtype=np.int32).astype(dtype)
    arr[8:24, 12:40] = hole
    return arr


def test_a_tile_without_holes_never_reaches_the_codec():
    """The regression that matters. No missing samples, no mask stream, so the
    frame has to be exactly the one this tile produced before nodata existed."""
    arr = _tile()
    assert _2d._nodata_args(arr, None) == (_2d._NODATA_NONE, 0.0)
    assert _2d._nodata_args(arr.astype(np.float32), None) == \
        (_2d._NODATA_NONE, 0.0)


def test_nan_only_triggers_on_float():
    assert _2d._nodata_args(_holed(), None) == (_2d._NODATA_NAN, 0.0)
    # an integer tile has no NaN to find, whatever it holds
    assert _2d._nodata_args(_tile(), None) == (_2d._NODATA_NONE, 0.0)


def test_a_declared_sentinel_wins_over_the_automatic_path():
    arr = _holed()
    assert _2d._nodata_args(arr, -9999) == (_2d._NODATA_VALUE, -9999.0)


def test_nodata_needs_a_dtype_geozl_knows():
    arr = np.zeros((4, 4), np.bool_)
    with pytest.raises(ValueError, match="nodata needs a dtype"):
        geozl.compress(arr, method=GRAPH_1B, nodata=1)


def test_nodata_survives_a_round_trip_through_the_2d_api():
    arr = _holed(np.int32, hole=-9999)
    out = _roundtrip(arr, method=GRAPH, nodata=-9999)
    assert np.array_equal(out, arr)


def test_profile_and_compress_agree_when_the_tile_holds_nan():
    """profile picks a recipe by timing it, so it has to time the graph compress
    will actually build. Without nodata reaching the bench the two disagree."""
    arr = _holed()
    row = geozl.profile(arr, reps=1)[0]
    frame = geozl.compress(arr, method=row["graph"])
    assert len(frame) == pytest.approx(arr.nbytes / row["ratio"], rel=0.01)


def test_profile_takes_a_sentinel_too():
    arr = _holed(np.int32, hole=-9999)
    row = geozl.profile(arr, reps=1, nodata=-9999)[0]
    frame = geozl.compress(arr, method=row["graph"], nodata=-9999)
    assert len(frame) == pytest.approx(arr.nbytes / row["ratio"], rel=0.01)


def test_profile_rejects_a_sentinel_on_a_dtype_geozl_cannot_read():
    with pytest.raises(ValueError, match="nodata needs a dtype"):
        geozl.profile(np.zeros((4, 4), np.bool_), nodata=1)