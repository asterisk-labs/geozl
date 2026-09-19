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


def _frame(arr, **kw):
    return geozl.compress(arr, graph=geozl.graph(arr, **kw))


def _roundtrip(arr, **kw):
    frame = _frame(arr, **kw)
    out = geozl.decompress(frame).view(arr.dtype).reshape(arr.shape)
    return frame, out


def test_nan_round_trips_with_its_payload():
    tile = _smooth(np.float32)
    tile[_holes()] = ODD_NAN
    _, out = _roundtrip(tile, method=GRAPH_WIDE)
    # Bitwise, since NaN != NaN and array_equal would wave the payload through.
    assert np.array_equal(out.view(np.uint32), tile.view(np.uint32))


def test_a_second_nan_payload_is_a_hole_too():
    """Matching bits instead of testing for NaN left every payload but the
    first for whatever ran next, and a quantizer has no answer for one."""
    tile = _smooth(np.float32)
    holes = _holes()
    tile[holes] = ODD_NAN
    other = np.array(0x7FA00042, dtype=np.uint32).view(np.float32)
    tile[3, 3] = other
    _, out = _roundtrip(tile, method=GRAPH_WIDE, error="LINEAR:MAX_ERROR=2")
    assert np.isnan(out[3, 3])
    assert np.isnan(out[holes]).all()
    assert not np.isnan(out[0, 0])


def test_a_clean_tile_round_trips():
    """Declaring a sentinel the tile does not contain is a mask of all valid.
    It costs a stream that codes to nothing, and it still round trips."""
    tile = _smooth(np.uint16)
    frame, out = _roundtrip(tile, method=GRAPH, nodata=40000)
    assert np.array_equal(out, tile)
    assert len(frame) < 2 * len(_frame(tile, method=GRAPH))


def test_tile_that_is_all_holes_round_trips():
    """No shape of its own. The fill leaves both streams constant and the
    backends collapse them."""
    tile = np.full((ROWS, COLS), -9999, dtype=np.int32)
    frame, out = _roundtrip(tile, method=GRAPH_WIDE, nodata=-9999)
    assert np.array_equal(out, tile)
    assert len(frame) < 400


def test_all_nan_tile_round_trips():
    tile = np.full((ROWS, COLS), ODD_NAN, dtype=np.float32)
    _, out = _roundtrip(tile, method=GRAPH_WIDE)
    assert np.array_equal(out.view(np.uint32), tile.view(np.uint32))


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
    with_codec = len(_frame(tile, method=GRAPH_WIDE, nodata=-9999.0))
    without = len(_frame(tile, method=GRAPH_WIDE))
    assert with_codec < without * 1.5


def test_an_undeclared_tile_without_nan_gets_no_codec():
    """The node is only in the graph when a mode is set, and nothing sets one
    for a float raster that holds no NaN."""
    tile = _smooth(np.float32)
    assert _2d._nodata_args(tile, None) == (_2d._NODATA_NONE, 0)


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
    # A dtype geozl has no code for cannot carry a sentinel.
    with pytest.raises(ValueError):
        geozl.graph(tile.view(np.dtype("V4")), method=GRAPH_WIDE, nodata=1)


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


def test_the_degenerate_tiles_go_through_the_one_wire_shape():
    """Both ends of the range, no holes at all and nothing but holes."""
    node = geozl.lossless.Nodata(COLS, value=-9999, dtype=np.int32)
    clean = _smooth(np.int32)
    assert np.array_equal(_low_level_roundtrip(node, clean), clean.reshape(-1))

    empty = np.full((ROWS, COLS), -9999, dtype=np.int32)
    assert np.array_equal(_low_level_roundtrip(node, empty), empty.reshape(-1))


def test_python_and_c_pack_the_same_header():
    """Cross reader. The Python node and the C binding write the same bytes, so
    a frame from one decodes in the other."""
    from geozl.lossless import nodata as _nd
    assert _nd.nodata_bits(-9999, np.int32) == 0xFFFFD8F1
    assert _nd.nodata_bits(np.float32(-9999.0), np.float32) == 0xC61C3C00
    assert _nd._pattern_bytes(0xFFFFD8F1, 4) == b"\xf1\xd8\xff\xff"
    assert _nd._pattern_bytes(0x7FC0BEEF, 4) == b"\xef\xbe\xc0\x7f"


def test_a_sentinel_needs_a_type_that_can_carry_one():
    # the two nodata_bits in 2d.c refuses, so both readers agree on the same
    # rasters
    with pytest.raises(ValueError, match="half float"):
        geozl.lossless.Nodata(COLS, value=1, dtype=np.float16)
    with pytest.raises(ValueError, match="no code for dtype"):
        geozl.lossless.Nodata(COLS, value=1, dtype=np.complex64)


# Sentinel guards around lossy stages.

def _bits(a):
    return a.view(f"u{a.itemsize}")


def _sentinel_bits(value, dtype):
    return np.array(value, dtype=dtype).view(f"u{np.dtype(dtype).itemsize}")


def _lossy(arr, sentinel, error, method="planar>zigzag>transpose>entropy"):
    valid = arr[_bits(arr) != _sentinel_bits(sentinel, arr.dtype)]
    g = geozl.graph(valid, method, width=arr.shape[1], planes=1, error=error,
                    nodata=sentinel)
    frame = geozl.compress(arr, graph=g)
    return frame, geozl.decompress(frame).view(arr.dtype).reshape(arr.shape)


def _bound(error, x):
    x = x.astype(np.float64)
    if isinstance(error, (int, float)):
        return np.full_like(x, float(error))
    if error.startswith("LINEAR"):
        return np.full_like(x, float(error.split("=")[1].split(",")[0]))
    if error.startswith("LOG") or error.endswith("%"):
        pct = float(error.split("=")[-1].rstrip("%")) if "=" in error \
            else float(error.rstrip("%"))
        return pct / 100.0 * np.abs(x)
    fields = dict(kv.split("=") for kv in error.split(":")[1].split(","))
    k = float(fields["MAX_ERROR"].rstrip("N"))
    return k * np.sqrt(float(fields["A"]) + float(fields["B"]) * x)


def _rng():
    return np.random.default_rng(7)


def _dark_u16():
    a = _rng().integers(1, 12, (ROWS, COLS)).astype(np.uint16)
    a[_holes()] = 0
    return a


def _near_top_u16():
    a = np.resize(np.arange(65400, 65535, dtype=np.uint16), (ROWS, COLS))
    a[_holes()] = 65535
    return a


def _near_bottom_i16():
    a = np.resize(np.arange(-32767, -32600, dtype=np.int16), (ROWS, COLS))
    a[_holes()] = -32768
    return a


def _both_sides_i16():
    a = _rng().integers(-50, 50, (ROWS, COLS)).astype(np.int16)
    a[_holes()] = 0
    return a


def _around_zero_f32():
    a = _rng().uniform(-0.4, 0.4, (ROWS, COLS)).astype(np.float32)
    a[_holes()] = 0.0
    return a


def _around_hundred_u16():
    a = _rng().integers(80, 121, (ROWS, COLS)).astype(np.uint16)
    a[_holes()] = 100
    return a


def _both_sides_f64():
    a = _rng().uniform(-10001.0, -9997.0, (ROWS, COLS))
    a[_holes()] = -9999.0
    return a


_COLLISIONS = [
    ("u16 zero, LINEAR", _dark_u16, 0, 2),
    ("u16 zero, LINEAR wide", _dark_u16, 0, 5),
    ("u16 zero, LINEAR values", _dark_u16, 0, "LINEAR:MAX_ERROR=5,STORE=VALUES"),
    ("u16 zero, LOG", _dark_u16, 0, "LOG:MAX_ERROR=20%"),
    ("u16 zero, SQRT", _dark_u16, 0, "SQRT:MAX_ERROR=1N,A=1,B=1"),
    ("u16 top, LINEAR", _near_top_u16, 65535, 8),
    ("u16 top, LINEAR wide", _near_top_u16, 65535, 64),
    ("i16 bottom, LINEAR", _near_bottom_i16, -32768, 8),
    ("i16 inside the data", _both_sides_i16, 0, 5),
    ("f32 zero", _around_zero_f32, 0.0, 0.5),
    ("f64 inside the data", _both_sides_f64, -9999.0, 1.0),
    ("u16 inside, LOG", _around_hundred_u16, 100, "LOG:MAX_ERROR=5%"),
    ("u16 inside, SQRT", _around_hundred_u16, 100, "SQRT:MAX_ERROR=1N,A=1,B=1"),
]


@pytest.mark.parametrize("make,sentinel,error",
                         [c[1:] for c in _COLLISIONS],
                         ids=[c[0] for c in _COLLISIONS])
def test_no_valid_sample_decodes_to_the_sentinel(make, sentinel, error):
    arr = make()
    _, out = _lossy(arr, sentinel, error)
    s = _sentinel_bits(sentinel, arr.dtype)
    hole = _bits(arr) == s
    assert hole.any() and (~hole).any()
    assert (_bits(out)[hole] == s).all()
    assert not (_bits(out)[~hole] == s).any()
    x = arr[~hole].astype(np.float64)
    err = np.abs(out[~hole].astype(np.float64) - x)
    assert (err <= _bound(error, arr[~hole])).all(), err.max()


def test_the_guard_actually_fires():
    arr = _dark_u16()
    _, out = _lossy(arr, 0, 5)
    hole = arr == 0
    # 1 to 2 rebuild as 0 on a step of 10; the guard turns them into 1.
    assert ((arr[~hole] <= 2) & (out[~hole] == 1)).sum() > 100


def test_the_other_signed_zero_comes_back_exactly():
    arr = _smooth(np.float32)
    arr[_holes()] = 0.0
    arr[::5, ::3] = -0.0
    _, out = _lossy(arr, 0.0, "LOG:MAX_ERROR=5%")
    neg_zero = _bits(arr) == np.uint32(0x80000000)
    assert neg_zero.any()
    assert (_bits(out)[neg_zero] == np.uint32(0x80000000)).all()


def test_an_undeclared_nan_is_kept_off_the_sentinel():
    arr = _smooth(np.float32)
    arr[_holes()] = 0.0
    arr[2, 2] = np.nan
    _, out = _lossy(arr, 0.0, 0.5)
    assert _bits(out)[2, 2] != 0
    assert _bits(out)[2, 2] == 1  # the smallest positive subnormal


def _nodata_header(frame):
    from openzl import ext as zl

    from geozl.lossless import _DECODERS as lossless_decoders
    from geozl.lossless.nodata import NodataDecoder
    from geozl.lossy import _DECODERS as lossy_decoders

    seen = []

    class Spy(NodataDecoder):
        def decode(self, state):
            seen.append(bytes(state.codec_header))
            super().decode(state)

    d = zl.DCtx()
    for decoder in lossless_decoders + lossy_decoders:
        d.register_custom_decoder(
            Spy() if decoder is NodataDecoder else decoder())
    d.decompress(frame)
    assert len(seen) == 1
    return seen[0]


def test_lossless_sentinel_is_guarded_too():
    tile = _smooth(np.uint16)
    tile[_holes()] = 7
    header = _nodata_header(_frame(tile, method=GRAPH, nodata=7))
    assert header == bytes([7, 0, 8, 0, 6, 0, 7, 0])


def test_lossy_nan_mode_keeps_the_plain_form():
    tile = _smooth(np.float32)
    tile[_holes()] = ODD_NAN
    frame = _frame(tile, method=GRAPH_WIDE, error="LINEAR:MAX_ERROR=2")
    assert _nodata_header(frame) == np.uint32(0x7FC0BEEF).tobytes()


def test_lossy_sentinel_writes_the_guarded_form():
    frame, _ = _lossy(_dark_u16(), 0, 2)
    assert _nodata_header(frame) == bytes([0, 0, 1, 0, 0, 0, 0, 0])
    frame, _ = _lossy(_both_sides_f64(), -9999.0, 1.0)
    s = np.float64(-9999.0)
    assert _nodata_header(frame) == b"".join([
        s.tobytes(), np.nextafter(s, np.inf).tobytes(),
        np.nextafter(s, -np.inf).tobytes(), s.tobytes()])


def test_low_level_nan_mode_takes_the_plain_form():
    tile = _smooth(np.float32)
    tile[_holes()] = ODD_NAN
    node = geozl.lossless.Nodata(COLS)
    out = _low_level_roundtrip(node, tile)
    assert np.array_equal(out.view(np.uint32), tile.reshape(-1).view(np.uint32))
    assert len(_nodata_header(_frame_from_node(tile, node))) == 4


def test_low_level_guard_refuses_a_stream_of_another_width():
    node = geozl.lossless.Nodata(COLS, value=0, dtype=np.uint16)
    with pytest.raises(Exception, match="dtype code"):
        _low_level_roundtrip(node, _smooth(np.uint8))


def test_low_level_guarded_node_round_trips_losslessly():
    tile = _both_sides_i16()
    node = geozl.lossless.Nodata(COLS, value=0, dtype=np.int16)
    assert np.array_equal(_low_level_roundtrip(node, tile), tile.reshape(-1))


def _frame_from_node(tile, node):
    zl = pytest.importorskip("openzl.ext")
    c = zl.Compressor()
    backend = zl.graphs.Compress()(c)
    pred = geozl.lossless.PlanarZigzag(COLS)(c, backend)
    c.select_starting_graph(node(c, pred, backend))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    return bytes(cc.compress(
        [zl.Input(zl.Type.Numeric, np.ascontiguousarray(tile).reshape(-1))]))


def _frame_size(tile, node):
    return len(_frame_from_node(tile, node))


def test_a_radius_keeps_the_guarded_mask_as_cheap_as_the_plain_one():
    zl = pytest.importorskip("openzl.ext")
    from geozl.lossless import nodata as _nd

    def node(dtype_code, radius):
        def build(compressor, values_successor, mask_successor):
            succ = [s if isinstance(s, zl.GraphID) else s.parameterize(compressor)
                    for s in (values_successor, mask_successor)]
            enc = _nd._Encoder(COLS, 0, dtype_code, radius)
            return compressor.build_static_graph(
                compressor.register_custom_encoder(enc), succ, name="nd")

        return build

    tile = _both_sides_i16()
    plain = _frame_size(tile, node(None, None))          # the unguarded form
    zero = _frame_size(tile, node(5, 0))                 # a lossless radius
    every = _frame_size(tile, node(5, None))             # every side recorded
    assert zero - plain <= 16
    assert every > zero + 100


def _level_with_the_widest_preimage(ramp, error, near):
    g = geozl.graph(ramp, "planar>zigzag>transpose>zstd" if ramp.itemsize > 2
                    else "planar>zigzag>transpose>entropy",
                    width=ramp.shape[1], planes=1, error=error)
    back = geozl.decompress(geozl.compress(ramp, graph=g)).view(ramp.dtype)
    levels, counts = np.unique(back, return_counts=True)
    order = np.argsort(np.abs(levels.astype(np.float64) - near))[:9]
    return levels[order][np.argmax(counts[order])], back.reshape(ramp.shape)


_RADIUS = [
    ("u16 LINEAR", np.uint16, 5, 1000, 1.0),
    ("i16 LINEAR at zero", np.int16, 3, 0, 1.0),
    ("f32 LINEAR at zero", np.float32, 0.5, 0.0, 0.01),
    ("u16 LOG", np.uint16, "LOG:MAX_ERROR=5%", 100, 1.0),
    ("f32 LOG", np.float32, "LOG:MAX_ERROR=1%", 50.0, 0.01),
    ("u16 SQRT", np.uint16, "SQRT:MAX_ERROR=1N,A=1,B=1", 100, 1.0),
]


@pytest.mark.parametrize("dtype,error,near,unit", [c[1:] for c in _RADIUS],
                         ids=[c[0] for c in _RADIUS])
def test_the_radius_covers_every_sample_a_quantizer_lands_on_the_sentinel(
        dtype, error, near, unit):
    span = 64 * unit
    ramp = (near + np.linspace(-span, span, ROWS * COLS)).astype(dtype)
    ramp = ramp.reshape(ROWS, COLS)
    s, plain = _level_with_the_widest_preimage(ramp, error, near)
    s_bits = _sentinel_bits(s, dtype)
    landed = (plain == s) & (_bits(ramp) != s_bits)
    assert (ramp[landed] < s).any() and (ramp[landed] > s).any()

    tile = ramp.copy()
    tile[_holes()] = s
    _, out = _lossy(tile, s.item(), error)
    hole = _bits(tile) == s_bits
    assert (_bits(out)[hole] == s_bits).all()
    assert not (_bits(out)[~hole] == s_bits).any()
    x = tile[~hole].astype(np.float64)
    err = np.abs(out[~hole].astype(np.float64) - x)
    assert (err <= _bound(error, tile[~hole])).all(), err.max()
