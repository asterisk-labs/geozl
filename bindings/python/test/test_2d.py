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


def _ftile(shape=(32, 32), lo=1e-6, hi=1e6, dtype=np.float32):
    """Log spaced and positive, the shape the warped curves are for. A narrow
    tile would not tell a fixed tolerance from one that follows the value."""
    rng = np.random.default_rng(1)
    n = shape[0] * shape[1]
    v = 10.0 ** np.linspace(np.log10(lo), np.log10(hi), n)
    return (v * rng.uniform(0.9, 1.1, n)).reshape(shape).astype(dtype)


def _frame(arr, **kw):
    return geozl.compress(arr, graph=geozl.graph(arr, **kw))


def _roundtrip(arr, **kw):
    return geozl.decompress(_frame(arr, **kw)).view(arr.dtype).reshape(arr.shape)


def test_rejects_non_native_byte_order():
    with pytest.raises(ValueError, match="native byte order"):
        geozl.graph(_tile().astype(">i2"), method=GRAPH)


def test_rejects_1d_without_width():
    with pytest.raises(ValueError, match="width"):
        geozl.graph(np.arange(64, dtype=np.int16), method=GRAPH)


def test_1d_with_width_encodes_like_the_2d_tile():
    arr = _tile()
    flat = _frame(arr.reshape(-1), method=GRAPH, width=arr.shape[1])
    assert flat == _frame(arr, method=GRAPH)


def test_rejects_an_element_width_openzl_cannot_stream():
    # complex128 is 16 bytes, a numeric stream is 1, 2, 4 or 8
    with pytest.raises(ValueError, match="bytes per element"):
        geozl.graph(np.zeros((4, 4), np.complex128), method=GRAPH)


def test_method_is_required():
    with pytest.raises(TypeError):
        geozl.graph(_tile())


@pytest.mark.parametrize("bad", [None, "", 123, b"planar>zigzag>entropy"])
def test_method_must_be_a_recipe_string(bad):
    with pytest.raises(ValueError, match="must be a recipe name"):
        geozl.graph(_tile(), method=bad)


def test_unknown_recipe_is_rejected_by_the_library():
    with pytest.raises(RuntimeError, match="unknown method"):
        geozl.graph(_tile(), method="nope>zigzag>entropy")


def test_recipe_that_does_not_fit_the_element_width_fails():
    # the point of dropping the search: no silent fallback to something else
    with pytest.raises(RuntimeError, match="1-byte elements"):
        geozl.graph(_tile(dtype=np.uint8), method=GRAPH)


def test_compress_is_deterministic():
    arr = _tile()
    assert _frame(arr, method=GRAPH) == _frame(arr, method=GRAPH)


def test_compress_actually_compresses():
    arr = _tile()
    assert len(_frame(arr, method=GRAPH)) < arr.nbytes


def test_no_error_is_lossless():
    arr = _tile()
    assert np.array_equal(_roundtrip(arr, method=GRAPH, error=None), arr)


# A number used to mean an absolute bound, and 0 or less meant lossless. Both
# readings are gone: the bound picks the curve, so it has to say which one.
@pytest.mark.parametrize("error", [0, -1, 1, 2.0])
def test_a_bare_number_is_not_an_error_recipe(error):
    with pytest.raises(ValueError, match="recipe string"):
        geozl.graph(_tile(), method=GRAPH, error=error)


def test_graph_rejects_a_dtype_no_quantizer_has_a_kernel_for():
    with pytest.raises(ValueError, match="do not support"):
        geozl.graph(np.zeros((4, 4), np.bool_), method=GRAPH_1B,
                    error="LINEAR:MAX_ERROR=2")


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint32, np.float64])
def test_round_trip_is_bit_exact(dtype):
    arr = _tile(dtype=np.int32).astype(dtype)
    graph = GRAPH_1B if arr.dtype.itemsize == 1 else GRAPH
    out = _roundtrip(arr, method=graph)
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


def test_decompress_returns_flat_bytes():
    # the frame names its codecs but not the caller's type or shape, so
    # decompress hands back the bytes and the caller finishes the job
    arr = _tile()
    back = geozl.decompress(_frame(arr, method=GRAPH))
    assert back.dtype == np.uint8
    assert back.ndim == 1
    assert back.nbytes == arr.nbytes
    assert np.array_equal(back.view(arr.dtype).reshape(arr.shape), arr)


def test_unreadable_frame_is_reported():
    with pytest.raises(RuntimeError, match="unreadable frame"):
        geozl.decompress(b"not a frame at all")


def test_corrupt_payload_is_reported():
    frame = bytearray(_frame(_tile(), method=GRAPH))
    frame[-8] ^= 0xFF  # past the header, so the size still reads
    with pytest.raises(RuntimeError, match="decompress failed"):
        geozl.decompress(bytes(frame))


def test_max_output_size_refuses_before_allocating():
    # the declared size is the frame's word, so untrusted bytes need a ceiling
    arr = _tile()
    frame = _frame(arr, method=GRAPH)
    with pytest.raises(ValueError, match="above the .* allowed"):
        geozl.decompress(frame, max_output_size=arr.nbytes - 1)


def test_max_output_size_allows_an_exact_fit():
    arr = _tile()
    out = geozl.decompress(_frame(arr, method=GRAPH), max_output_size=arr.nbytes)
    assert np.array_equal(out.view(arr.dtype).reshape(arr.shape), arr)


def test_no_ceiling_is_the_default():
    arr = _tile()
    assert geozl.decompress(_frame(arr, method=GRAPH)).nbytes == arr.nbytes


def test_verification_off_still_round_trips():
    arr = _tile()
    out = geozl.decompress(_frame(arr, method=GRAPH), verify=False)
    assert np.array_equal(out.view(arr.dtype).reshape(arr.shape), arr)


def test_verification_is_what_reads_the_checksums():
    """Both checksums sit in the frame tail. Every flip there is refused with
    verification on. With it off some come back untouched, which are the ones
    that landed on a checksum and nowhere else, and the rest come back holding a
    different raster without a word."""
    arr = _tile()
    frame = _frame(arr, method=GRAPH)
    intact = quiet = 0
    for pos in range(2, 64):
        bad = bytearray(frame)
        bad[-pos] ^= 0xFF
        bad = bytes(bad)
        with pytest.raises(RuntimeError, match="decompress failed"):
            geozl.decompress(bad)
        try:
            out = geozl.decompress(bad, verify=False).view(arr.dtype)
        except RuntimeError:
            continue
        if np.array_equal(out.reshape(arr.shape), arr):
            intact += 1
        else:
            quiet += 1
    assert intact and quiet


@pytest.mark.parametrize("recipe,bound",
                         [("LINEAR:MAX_ERROR=1", 1),
                          ("LINEAR:MAX_ERROR=2.0", 2.0),
                          ("LINEAR:MAX_ERROR=7", 7)])
def test_error_bound_holds(recipe, bound):
    arr = _tile()
    out = _roundtrip(arr, method=GRAPH, error=recipe)
    assert np.abs(out.astype(np.int64) - arr.astype(np.int64)).max() <= bound


def test_lossy_beats_lossless_on_size():
    arr = _tile()
    assert len(_frame(arr, method=GRAPH, error="LINEAR:MAX_ERROR=8")) < \
           len(_frame(arr, method=GRAPH))


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


def test_profile_reports_the_frame_compress_writes_to_the_byte():
    """Not approximately. The bench used to time with the content checksum off,
    which left its frame four bytes under the one compress writes."""
    arr = _tile()
    for row in geozl.profile(arr, prior=None, reps=1):
        frame = _frame(arr, method=row["graph"])
        assert len(frame) == row["bytes"], row["graph"]
        assert row["ratio"] == arr.nbytes / len(frame)


def test_unbiased_prior_sweeps_more_graphs_than_a_named_one():
    arr = _tile()
    assert len(geozl.profile(arr, prior=None, reps=1)) > \
           len(geozl.profile(arr, prior="planar", reps=1))


def test_unbiased_prior_is_the_whole_grid():
    # 8 predictors x 7 terminals, all buildable at 2 bytes per element
    assert len(geozl.profile(_tile(), prior=None, reps=1)) == 56


def test_the_grid_at_one_byte_keeps_categorical_and_drops_the_rest():
    # 8 x 5: the two transposed terminals need 2 bytes; categorical and pfor do
    # not.
    rows = geozl.profile(_tile(dtype=np.uint8), prior=None, reps=1)
    assert len(rows) == 40
    assert any(r["graph"].endswith("categorical") for r in rows)
    assert any(r["graph"].endswith("pfor") for r in rows)


def test_transpose_terminals_drop_out_on_1_byte_elements():
    rows = geozl.profile(_tile(dtype=np.uint8), prior=None, reps=1)
    assert rows
    assert not any("transpose" in r["graph"] for r in rows)


def test_store_lo_is_no_longer_a_recipe():
    with pytest.raises(RuntimeError, match="unknown method"):
        geozl.graph(_tile(), "planar>zigzag>store_lo")


def test_profile_rejects_an_unknown_prior():
    with pytest.raises(ValueError, match="not one of"):
        geozl.profile(_tile(), prior="not_a_predictor")


def test_categorical_drops_out_past_two_bytes():
    # its entropy arm tops out at 2, and a terminal that applies or not by what
    # the tile happens to hold would be worse than one that never applies
    rows = geozl.profile(_tile(dtype=np.uint32), prior=None, reps=1)
    assert rows and not any("categorical" in r["graph"] for r in rows)


def test_verify_moves_the_clock_and_never_the_frame():
    arr = _tile()
    on = geozl.profile(arr, prior=None, reps=1, verify=True)
    off = geozl.profile(arr, prior=None, reps=1, verify=False)
    assert {r["graph"]: r["bytes"] for r in on} == \
           {r["graph"]: r["bytes"] for r in off}


def test_profile_needs_at_least_one_rep():
    # reps=0 failed every bench call and came back as an empty table
    with pytest.raises(ValueError, match="at least 1"):
        geozl.profile(_tile(), reps=0)


def test_a_tile_too_small_to_time_still_profiles():
    """One rep over a kilobyte can finish inside a clock tick, and macOS ticks
    at a microsecond. A zero there used to divide."""
    rows = geozl.profile(np.zeros((8, 8), np.uint8), prior=None, reps=1)
    assert rows
    assert all(r["encode_mbps"] > 0 and r["decode_mbps"] > 0 for r in rows)


def test_profile_lossy_beats_profile_lossless():
    arr = _tile()
    assert geozl.profile(arr, reps=1, error="LINEAR:MAX_ERROR=8")[0]["ratio"] > \
           geozl.profile(arr, reps=1)[0]["ratio"]


def test_profile_rejects_a_dtype_no_quantizer_has_a_kernel_for():
    with pytest.raises(ValueError, match="do not support"):
        geozl.profile(np.zeros((4, 4), np.bool_), error="LINEAR:MAX_ERROR=2")


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
    """Bigger than _tile on purpose, so the mask stream weighs something against
    the raster."""
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
    # the float32 bits of -9999.0, not the number
    assert _2d._nodata_args(arr, -9999) == (_2d._NODATA_VALUE, 0xC61C3C00)


def test_a_nan_sentinel_takes_the_automatic_path():
    """As bits it would match one payload, not every NaN in the tile."""
    assert _2d._nodata_args(_holed(), float("nan")) == (_2d._NODATA_NAN, 0)


@pytest.mark.parametrize("dtype,hole", [
    (np.int32, -9999),          # the ordinary one, and the control
    (np.int64, 2 ** 53 - 1),    # last sentinel a double still held
    (np.int64, 2 ** 53 + 1),    # first one it dropped, with no overflow to see
    (np.int64, 2 ** 63 - 1),    # the signed end
    (np.uint64, 2 ** 64 - 1),   # the unsigned end, and the cast that was UB
])
def test_a_wide_sentinel_reaches_the_codec(dtype, hole):
    """A sentinel scattered through a smooth raster is filled away when it
    matches and left as a cliff when it misses, so the two sizes are the
    witness. Carried through a double the wide ones would miss."""
    n = 128
    arr = _tile((n, n)).astype(dtype)
    arr.reshape(-1)[::7] = dtype(hole)
    matched = _frame(arr, method=GRAPH, nodata=hole)
    # 1000 is past the ramp, so the miss really misses
    missed = _frame(arr, method=GRAPH, nodata=dtype(1000))
    assert len(matched) * 2 < len(missed)
    assert np.array_equal(
        geozl.decompress(matched).view(arr.dtype).reshape(n, n), arr)


def test_a_sentinel_outside_the_dtype_is_refused():
    arr = _tile(shape=(8, 8), dtype=np.int32).astype(np.uint64)
    with pytest.raises(OverflowError):
        geozl.graph(arr, method=GRAPH, nodata=-1)


def test_a_fractional_sentinel_on_an_integer_tile_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        geozl.graph(_tile(), method=GRAPH, nodata=3.5)


def test_nodata_needs_a_dtype_geozl_knows():
    arr = np.zeros((4, 4), np.bool_)
    with pytest.raises(ValueError, match="nodata needs a dtype"):
        geozl.graph(arr, method=GRAPH_1B, nodata=1)


def test_nodata_survives_a_round_trip_through_the_2d_api():
    arr = _holed(np.int32, hole=-9999)
    out = _roundtrip(arr, method=GRAPH, nodata=-9999)
    assert np.array_equal(out, arr)


def test_profile_and_compress_agree_when_the_tile_holds_nan():
    """profile picks a recipe by timing it, so it has to time the graph compress
    will actually build. Without nodata reaching the bench the two disagree."""
    arr = _holed()
    row = geozl.profile(arr, reps=1)[0]
    assert len(_frame(arr, method=row["graph"])) == row["bytes"]


def test_profile_takes_a_sentinel_too():
    arr = _holed(np.int32, hole=-9999)
    row = geozl.profile(arr, reps=1, nodata=-9999)[0]
    frame = _frame(arr, method=row["graph"], nodata=-9999)
    assert len(frame) == row["bytes"]


def test_profile_rejects_a_sentinel_on_a_dtype_geozl_cannot_read():
    with pytest.raises(ValueError, match="nodata needs a dtype"):
        geozl.profile(np.zeros((4, 4), np.bool_), nodata=1)

# Through the high level entry, which fits a SQRT recipe with no curve and
# scans the whole raster. A node placed by hand is in test_codecs.py.


@pytest.mark.parametrize("pct", [0.5, 1.0, 10.71])
@pytest.mark.parametrize("dtype", [np.float32, np.float64],
                         ids=lambda d: np.dtype(d).name)
def test_relative_bound_holds_through_compress(pct, dtype):
    arr = _ftile(dtype=dtype)
    out = _roundtrip(arr, method=GRAPH, error=f"LOG:MAX_ERROR={pct}%")
    x = arr.astype(np.float64)
    assert (np.abs(out.astype(np.float64) - x) <= (pct / 100.0) * x).all()


def test_relative_bound_does_not_anchor_on_the_tile():
    # The anchor comes from the type and the bound that was asked for, so adding
    # one sample below everything else must not move it. Anchoring on the
    # smallest magnitude present buys a shorter index, but every level of the log
    # grid is a multiple of the anchor, so one outlying sample would then change
    # the reconstruction of every other sample in the tile, and the same value
    # would stop reconstructing the same way once the raster is cut differently.
    arr = _ftile()
    low = arr.copy()
    low.reshape(-1)[0] = np.float32(1e-20)  # far under the rest of the tile
    a = _roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=1%").reshape(-1)
    b = _roundtrip(low, method=GRAPH, error="LOG:MAX_ERROR=1%").reshape(-1)
    assert np.array_equal(a[1:], b[1:])


def test_a_non_negative_tile_decodes_to_nothing_negative():
    # The sqrt grid is anchored at -offset rather than at zero, so without the
    # floor a tile of non-negative data comes back holding small negatives, well
    # inside the bound and still wrong for a reflectance or a backscatter. The
    # bound alone does not catch it, since the bound at zero is wide enough to
    # cover the crossing.
    arr = _ftile(lo=1e-6, hi=1.0)
    arr.reshape(-1)[:16] = 0.0
    out = _roundtrip(arr, method=GRAPH,
                     error="SQRT:MAX_ERROR=0.5N,A=1e-12,B=1e-6").reshape(-1)
    assert out.min() >= 0.0
    assert np.all(out[:16] == 0.0)  # and an exact zero comes back exact


def test_relative_bound_gives_zero_back_exactly():
    arr = _ftile()
    arr[0] = 0.0
    out = _roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=1%")
    assert (out.reshape(arr.shape)[0] == 0.0).all()


def test_relative_bound_on_an_all_zero_tile():
    arr = np.zeros((32, 32), dtype=np.float32)
    assert np.array_equal(_roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=1%"), arr)


def test_shot_bound_holds_through_compress():
    arr = np.linspace(0.0, 10000.0, 1024, dtype=np.float32).reshape(32, 32)
    out = _roundtrip(arr, method=GRAPH, error="SQRT:MAX_ERROR=0.5N,A=4,B=1")
    x = arr.astype(np.float64)
    assert (np.abs(out.astype(np.float64) - x) <= 0.5 * np.sqrt(4.0 + x)).all()


# Optical reflectance arrives as unsigned integers, and near the bottom of the
# range the sqrt curve reconstructs below zero, which is where a missing clamp
# would wrap instead of saturating.
def test_shot_bound_holds_on_an_unsigned_raster():
    arr = (np.arange(1024, dtype=np.uint16) * 5).reshape(32, 32)
    out = _roundtrip(arr, method=GRAPH, error="SQRT:MAX_ERROR=0.5N,A=100,B=1")
    x = arr.astype(np.float64)
    assert (np.abs(out.astype(np.float64) - x) <= 0.5 * np.sqrt(100.0 + x)).all()


# A relative bound is symmetric, so the sign has to survive the trip.
def test_relative_bound_keeps_the_sign():
    arr = _ftile()
    arr[::3] *= -1.0
    out = _roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=1%").reshape(arr.shape)
    assert (np.sign(out) == np.sign(arr)).all()
    x = arr.astype(np.float64)
    assert (np.abs(out.astype(np.float64) - x) <= 0.01 * np.abs(x)).all()


# Wide enough that the index range leaves the reconstruction table and the
# decoder falls back to evaluating the curve per sample.
def test_relative_bound_beyond_the_reconstruction_table():
    arr = _ftile()
    out = _roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=0.01%")
    x = arr.astype(np.float64)
    assert (np.abs(out.astype(np.float64) - x) <= 0.0001 * x).all()


def test_shot_refuses_a_raster_reaching_below_the_anchor():
    arr = _ftile()
    arr[0] = -10.0  # under the -A/B of -4 where the curve stops being defined
    with pytest.raises(RuntimeError, match="at or above"):
        geozl.graph(arr, method=GRAPH, error="SQRT:MAX_ERROR=0.5N,A=4,B=1")


def test_a_bound_finer_than_the_output_type_is_refused():
    with pytest.raises(RuntimeError, match="what this type rebuilds to"):
        geozl.graph(_ftile(), method=GRAPH, error="LOG:MAX_ERROR=1e-7%")


def test_a_relative_bound_no_grid_can_serve_is_lossless_not_wrong():
    # On integers the representable values sit one apart, so below roughly 8/b
    # nothing but the sample itself is inside the bound and the codec carries
    # that range exactly. A tight bound on uint16 puts the whole type in there,
    # which costs ratio but never the bound.
    arr = _tile(dtype=np.uint16)
    assert np.array_equal(_roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=0.01%"), arr)


# nodata runs in front of the quantizer but the scan it needs reads
# the tile before it, so what nodata substitutes has to already be inside the
# grid. It fills with the last valid sample of the row, or the one above, and
# falls back to zero only when the hole leads the first row. These are those.
@pytest.mark.parametrize("row", [0, 3], ids=["leading_row", "inner_row"])
def test_relative_bound_keeps_nodata_exact(row):
    arr = _ftile()
    arr[row, :] = -9999.0
    out = _roundtrip(arr, method=GRAPH, error="LOG:MAX_ERROR=1%", nodata=-9999.0)
    out = out.reshape(arr.shape)
    assert (out[row] == -9999.0).all()
    keep = np.ones(arr.shape, bool)
    keep[row] = False
    x = arr[keep].astype(np.float64)
    assert (np.abs(out[keep].astype(np.float64) - x) <= 0.01 * np.abs(x)).all()


@pytest.mark.parametrize("recipe", ["LOG:MAX_ERROR=1%",
                                   "SQRT:MAX_ERROR=0.5N,A=4,B=1"])
def test_profile_accepts_the_warped_recipes(recipe):
    rows = geozl.profile(_ftile(), reps=1, error=recipe)
    assert rows and all(r["ratio"] > 0 for r in rows)
