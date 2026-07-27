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

# entropy wants 1 or 2 bytes per element, field_lz takes any of the four, so the
# float tiles below go through field_lz.
GRAPH = "planar>zigzag>entropy"
GRAPH_WIDE = "planar>zigzag>field_lz"

ROWS, COLS = 48, 64

# A NaN carries 22 payload bits in f32. Storing the pattern rather than a
# canonical NaN is what keeps the round trip lossless, so the tests use one no
# hardware operation would produce.
ODD_NAN = np.array(0x7FC0BEEF, dtype=np.uint32).view(np.float32)


def _smooth(dtype, shape=(ROWS, COLS)):
    y, x = np.indices(shape)
    return (285.0 + 0.5 * x + 0.25 * y).astype(dtype)


def _holes(shape=(ROWS, COLS)):
    """A coherent blob plus a whole trailing row, which is the fill path that
    has no valid sample to its left and has to reach the row above."""
    y, x = np.indices(shape)
    return ((y - 16) ** 2 + (x - 20) ** 2 < 120) | (y == shape[0] - 1)


def _roundtrip(arr, **kw):
    frame = geozl.compress(arr, **kw)
    out = geozl.decompress(frame, dtype=arr.dtype.name, width=arr.shape[1])
    return frame, out


def test_nan_round_trips_with_its_payload():
    tile = _smooth(np.float32)
    tile[_holes()] = ODD_NAN
    _, out = _roundtrip(tile, method=GRAPH_WIDE)
    # Bitwise, since NaN != NaN and array_equal would wave the payload through.
    assert np.array_equal(out.view(np.uint32), tile.view(np.uint32))


def test_a_second_nan_payload_is_a_hole_too():
    """Matching bits instead of testing for NaN left every payload but the
    first for whatever ran next, and quant_linear has no answer for one."""
    tile = _smooth(np.float32)
    holes = _holes()
    tile[holes] = ODD_NAN
    other = np.array(0x7FA00042, dtype=np.uint32).view(np.float32)
    tile[3, 3] = other
    _, out = _roundtrip(tile, method=GRAPH_WIDE, max_error=2)
    assert np.isnan(out[3, 3])
    assert np.isnan(out[holes]).all()
    assert not np.isnan(out[0, 0])


def test_sentinel_round_trips():
    tile = _smooth(np.int32)
    tile[_holes()] = -9999
    _, out = _roundtrip(tile, method=GRAPH_WIDE, nodata=-9999)
    assert np.array_equal(out, tile)


def test_float_sentinel_round_trips():
    tile = _smooth(np.float32)
    tile[_holes()] = -9999.0
    _, out = _roundtrip(tile, method=GRAPH_WIDE, nodata=-9999.0)
    assert np.array_equal(out.view(np.uint32), tile.view(np.uint32))


@pytest.mark.parametrize("dtype", ["uint8", "int16", "uint16", "int32",
                                   "float32", "float64"])
def test_sentinel_every_dtype(dtype):
    tile = _smooth(dtype)
    tile[_holes()] = 7
    graph = GRAPH if np.dtype(dtype).itemsize <= 2 else GRAPH_WIDE
    _, out = _roundtrip(tile, method=graph, nodata=7)
    assert np.array_equal(out, tile)


def test_holes_do_not_blow_up_the_frame():
    tile = _smooth(np.float32)
    tile[_holes()] = -9999.0
    with_codec = len(geozl.compress(tile, method=GRAPH_WIDE, nodata=-9999.0))
    without = len(geozl.compress(tile, method=GRAPH_WIDE))
    assert with_codec < without * 1.5


def test_tile_without_holes_skips_the_codec():
    """No missing samples, no mask stream, so the frame has to match the one the
    same tile produced before the codec existed."""
    tile = _smooth(np.float32)
    assert _2d._nodata_args(tile, None) == (_2d._NODATA_NONE, 0.0)


def test_nan_detected_only_on_float():
    ints = _smooth(np.int32)
    assert _2d._nodata_args(ints, None) == (_2d._NODATA_NONE, 0.0)
    floats = _smooth(np.float32)
    floats[0, 0] = np.nan
    assert _2d._nodata_args(floats, None) == (_2d._NODATA_NAN, 0.0)


def test_infinity_is_a_value_not_a_hole():
    tile = _smooth(np.float32)
    tile[_holes()] = np.inf
    _, out = _roundtrip(tile, method=GRAPH_WIDE)
    assert np.array_equal(out.view(np.uint32), tile.view(np.uint32))


def test_sentinel_needs_a_known_dtype():
    tile = _smooth(np.float32).astype(np.dtype("f4"))
    # A dtype geozl has no quant_linear code for cannot carry a sentinel.
    with pytest.raises(ValueError):
        geozl.compress(tile.view(np.dtype("V4")), method=GRAPH_WIDE, nodata=1)


def _low_level_roundtrip(node, arr):
    """Same shape as the other codec tests, a node placed by hand in a graph."""
    zl = pytest.importorskip("openzl.ext")
    c = zl.Compressor()
    backend = zl.graphs.Compress()(c)
    c.select_starting_graph(node(c, backend, backend))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    frame = bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))

    d = zl.DCtx()
    geozl.register_decoders(d)
    return d.decompress(frame)[0].content.as_nparray().view(arr.dtype)


def test_low_level_node_takes_an_unbound_successor():
    zl = pytest.importorskip("openzl.ext")
    c = zl.Compressor()
    node = geozl.lossless.Nodata(COLS)
    assert isinstance(node(c, zl.graphs.Compress(), zl.graphs.Compress()),
                      zl.GraphID)


def test_low_level_nan_round_trips():
    """The codec placed by hand, which is what the README table advertises."""
    tile = _smooth(np.float32)
    tile[_holes()] = ODD_NAN
    out = _low_level_roundtrip(geozl.lossless.Nodata(COLS), tile)
    assert np.array_equal(out.view(np.uint32), tile.reshape(-1).view(np.uint32))


def test_low_level_sentinel_round_trips():
    tile = _smooth(np.int32)
    tile[_holes()] = -9999
    node = geozl.lossless.Nodata(COLS, value=-9999, dtype=np.int32)
    out = _low_level_roundtrip(node, tile)
    assert np.array_equal(out, tile.reshape(-1))


def test_low_level_sentinel_needs_a_dtype():
    with pytest.raises(ValueError, match="dtype"):
        geozl.lossless.Nodata(COLS, value=-9999)


def test_python_and_c_pack_the_same_header():
    """Cross reader. The Python node and the C binding write the same bytes, so
    a frame from one decodes in the other."""
    from geozl.lossless import nodata as _nd
    assert _nd.nodata_bits(-9999, np.int32) == 0xFFFFD8F1
    assert _nd.nodata_bits(np.float32(-9999.0), np.float32) == 0xC61C3C00
    assert _nd._pattern_bytes(0xFFFFD8F1, 4) == b"\xf1\xd8\xff\xff"
    assert _nd._pattern_bytes(0x7FC0BEEF, 4) == b"\xef\xbe\xc0\x7f"