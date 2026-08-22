from itertools import repeat

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _coeffs  # noqa: E402

try:
    _coeffs._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

_W = 64
ONE = ((23,),)
VECTORS = ((-7, 0, 11, 42), (3, 5, -9, 17))


@pytest.fixture(scope="module")
def tile():
    y, x = np.indices((_W, _W))
    return ((x * 7 + y * 13) % 9000 + 500).astype(np.uint16)


@pytest.fixture(scope="module")
def graph(tile):
    return geozl.graph(tile, method="planar>zigzag>pfor")


def test_absent(tile, graph):
    assert geozl.coeffs(geozl.compress(tile, graph=graph)) is None
    assert geozl.compress(tile, graph=graph, coeffs=None) == \
        geozl.compress(tile, graph=graph)


@pytest.mark.parametrize(
    "vectors", [ONE, VECTORS, ((0,),), ((1, 2, 3),)]
)
def test_round_trip(tile, graph, vectors):
    frame = geozl.compress(tile, graph=graph, coeffs=vectors)
    assert geozl.coeffs(frame) == vectors


def test_payload_is_unchanged(tile, graph):
    plain = geozl.compress(tile, graph=graph)
    tagged = geozl.compress(tile, graph=graph, coeffs=VECTORS)
    want = tile.reshape(-1)
    for frame in (plain, tagged):
        got = geozl.decompress(frame).view(tile.dtype)
        assert np.array_equal(got, want)


def test_coeffs_do_not_leak_between_compressions(tile, graph):
    before = geozl.compress(tile, graph=graph)
    tagged = geozl.compress(tile, graph=graph, coeffs=VECTORS)
    after = geozl.compress(tile, graph=graph)
    assert geozl.coeffs(tagged) == VECTORS
    assert geozl.coeffs(after) is None
    assert before == after


def test_failed_compression_clears_coeffs(tile, graph):
    base = geozl.compress(tile, graph=graph)
    odd = np.frombuffer(bytearray(tile.nbytes + 8), np.uint8)[1:]
    odd = odd[:tile.nbytes].view(tile.dtype)
    assert odd.ctypes.data % tile.dtype.itemsize != 0

    with pytest.raises(RuntimeError, match="numeric stream"):
        geozl.compress(odd, graph=graph, coeffs=VECTORS)

    after = geozl.compress(tile, graph=graph)
    assert geozl.coeffs(after) is None
    assert after == base


def test_extremes(tile, graph):
    edge = ((-2 ** 31, -1, 0, 1, 2 ** 31 - 1),)
    frame = geozl.compress(tile, graph=graph, coeffs=edge)
    assert geozl.coeffs(frame) == edge


def test_frame_overhead(tile, graph):
    plain = len(geozl.compress(tile, graph=graph))
    for vectors, blob in ((ONE, 6 + 4 + 4), (VECTORS, 6 + 2 * (4 + 16))):
        grew = len(geozl.compress(tile, graph=graph, coeffs=vectors)) - plain
        assert blob <= grew <= blob + 3


def test_output_capacity_includes_coeffs():
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    graph = geozl.graph(tile, method="id>zstd")
    vectors = ([-7] * 2497,)
    frame = geozl.compress(tile, graph=graph, coeffs=vectors)
    assert geozl.coeffs(frame) == (tuple(vectors[0]),)


@pytest.mark.parametrize("bad", [
    (),                       # no vectors
    ((),),                    # an empty vector
    ((1,), ()),               # one of them empty
    [[0] * 1249] * 2,         # past the comment ceiling
    [[0]] * 256,              # past 255 vectors
])
def test_invalid_shapes(tile, graph, bad):
    with pytest.raises(ValueError):
        geozl.compress(tile, graph=graph, coeffs=bad)


@pytest.mark.parametrize("bad", [(repeat(1),), repeat((1,))])
def test_unbounded_iterables_are_rejected(tile, graph, bad):
    with pytest.raises(ValueError):
        geozl.compress(tile, graph=graph, coeffs=bad)


def test_size_limit(tile, graph):
    with pytest.raises(ValueError, match="10000-byte"):
        geozl.compress(tile, graph=graph, coeffs=[[0] * 1249] * 2)


def test_python_limits_match_the_library(tile, graph):
    from geozl._ffi import ffi
    from geozl._ffi import lib as kernels

    largest = (_coeffs._MAX_BYTES - 6 - 4) // 4
    geozl.compress(tile, graph=graph, coeffs=[[0] * largest])
    with pytest.raises(ValueError):
        geozl.compress(tile, graph=graph, coeffs=[[0] * (largest + 1)])
    counts = ffi.new("uint32_t[]", [largest])
    assert kernels.geozl_coeffs_size(counts, 1) == _coeffs._MAX_BYTES - 2
    counts[0] = largest + 1
    assert kernels.geozl_coeffs_size(counts, 1) == 0
    geozl.compress(tile, graph=graph, coeffs=[[0]] * _coeffs._MAX_VECTORS)
    with pytest.raises(ValueError):
        geozl.compress(tile, graph=graph, coeffs=[[0]] * (_coeffs._MAX_VECTORS + 1))


@pytest.mark.parametrize("bad", [True, False])
def test_booleans_are_not_coefficients(tile, graph, bad):
    with pytest.raises(TypeError, match="boolean"):
        geozl.compress(tile, graph=graph, coeffs=[[bad]])


def test_maximum_blob(tile, graph):
    frame = geozl.compress(tile, graph=graph, coeffs=[[3] * 1248] * 2)
    got = geozl.coeffs(frame)
    assert len(got) == 2 and len(got[0]) == 1248 and got[1][1247] == 3


@pytest.mark.parametrize("bad", [
    "not a sequence", 3, [1, 2], {"scale": [1]},
    "12", b"\x01\x02", bytearray(b"ab"), ["12"], [b"\x01\x02"],
    [["12"]], [[b"\x01\x02"]],
    [[2.5]], [[1e-4]],
])
def test_invalid_types(tile, graph, bad):
    with pytest.raises((TypeError, ValueError)):
        geozl.compress(tile, graph=graph, coeffs=bad)


@pytest.mark.parametrize("bad", [2 ** 31, -(2 ** 31) - 1, 10 ** 400])
def test_value_outside_int32(tile, graph, bad):
    with pytest.raises(ValueError, match="int32"):
        geozl.compress(tile, graph=graph, coeffs=[[bad]])


def test_numpy_container(tile, graph):
    vectors = np.array([[7, -42], [1000, -2000]], np.int32)
    frame = geozl.compress(tile, graph=graph, coeffs=vectors)
    assert geozl.coeffs(frame) == ((7, -42), (1000, -2000))


def test_unreadable_frame(tile, graph):
    frame = bytearray(geozl.compress(tile, graph=graph, coeffs=ONE))
    frame[:4] = b"\x00\x00\x00\x00"
    with pytest.raises(RuntimeError, match="unreadable frame"):
        geozl.coeffs(bytes(frame))


def test_c_api_distinguishes_absence_from_failure(tile, graph):
    from geozl._ffi import _load_lib_full, _ptr, ffi
    full = _load_lib_full()
    size = ffi.new("size_t*")

    plain = np.frombuffer(geozl.compress(tile, graph=graph), np.uint8)
    assert full.geozl_2d_frame_coeffs_c(_ptr(plain), plain.size, ffi.NULL, 0,
                                        size) < 0

    tagged = np.frombuffer(geozl.compress(tile, graph=graph, coeffs=ONE),
                           np.uint8)
    assert full.geozl_2d_frame_coeffs_c(_ptr(tagged), tagged.size, ffi.NULL, 0,
                                        size) == 0
    assert size[0] == 6 + 4 + 4

    junk = np.frombuffer(b"\x00" * 32, np.uint8)
    assert full.geozl_2d_frame_coeffs_c(_ptr(junk), junk.size, ffi.NULL, 0,
                                        size) > 0


def test_c_api_rejects_empty_blob(tile, graph):
    from geozl._ffi import _load_lib_full, _ptr, ffi
    full = _load_lib_full()
    arr = np.ascontiguousarray(tile)
    dst = np.empty(1 << 16, np.uint8)
    out = ffi.new("size_t*")
    err = ffi.new("char[]", 256)
    rc = full.geozl_2d_compress_coeffs_c(
        graph._h, _ptr(arr), arr.size, ffi.new("char[]", 1), 0, _ptr(dst),
        dst.size, out, err, len(err))
    assert rc != 0
    assert b"valid blob" in ffi.string(err)

    rc = full.geozl_2d_compress_coeffs_c(
        graph._h, _ptr(arr), arr.size, ffi.NULL, 1, _ptr(dst), dst.size, out,
        err, len(err))
    assert rc != 0


def test_foreign_blob_is_absent():
    assert _coeffs._parse_coeffs(b"hello, world") is None
    assert _coeffs._parse_coeffs(b"") is None
    assert _coeffs._parse_coeffs(b"GZC" + b"\x00" * 20) is None


def test_invalid_geozl_blob_raises():
    for bad in (b"GZC1\x02\x01" + b"\x00" * 12,      # unknown version
                b"GZC1\x01\x00" + b"\x00" * 12,      # no vectors
                b"GZC1\x01\x01\x01\x00\x00\x00",      # count runs past the end
                b"GZC1\x01\x01\x01\x00\x00\x00" + b"\x00" * 9):  # trailing byte
        with pytest.raises(ValueError, match="invalid"):
            _coeffs._parse_coeffs(bad)


def test_a_frame_older_than_the_comment_field_is_absent(tile):
    # Header comments were added in frame format version 22.
    zl = pytest.importorskip("openzl.ext")
    compressor = zl.Compressor()
    compressor.select_starting_graph(zl.graphs.Compress().parameterize(compressor))
    cctx = zl.CCtx()
    cctx.ref_compressor(compressor)
    cctx.set_parameter(zl.CParam.FormatVersion, 20)
    frame = bytes(cctx.compress([zl.Input(zl.Type.Numeric, tile.reshape(-1))]))
    assert geozl.coeffs(frame) is None


def test_a_comment_from_another_producer(tile):
    zl = pytest.importorskip("openzl.ext")
    if not hasattr(zl.CCtx, "add_header_comment"):
        pytest.skip("openzl's python binding cannot write a frame comment")

    def framed(comment):
        compressor = zl.Compressor()
        compressor.select_starting_graph(
            zl.graphs.Compress().parameterize(compressor))
        cctx = zl.CCtx()
        cctx.ref_compressor(compressor)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        cctx.add_header_comment(comment)
        return bytes(cctx.compress([zl.Input(zl.Type.Numeric, tile.reshape(-1))]))

    # Foreign comments do not contain geozl coefficients.
    assert geozl.coeffs(framed(b"someone else's note")) is None
    # Matching magic with an invalid body is frame corruption.
    with pytest.raises(RuntimeError, match="unreadable frame"):
        geozl.coeffs(framed(b"GZC1\x01\x00"))


@pytest.mark.parametrize("kwargs", [
    {"error": 5},                       # a lossy graph
    {"nodata": 0},                      # a nodata graph
    {"width": 32, "planes": 4},         # several planes
])
def test_coeffs_are_orthogonal_to_the_graph(tile, kwargs):
    g = geozl.graph(tile, method="planar>zigzag>pfor", **kwargs)
    plain = geozl.compress(tile, graph=g)
    tagged = geozl.compress(tile, graph=g, coeffs=VECTORS)
    assert geozl.coeffs(plain) is None
    assert geozl.coeffs(tagged) == VECTORS
    assert geozl.decompress(plain).tobytes() == geozl.decompress(tagged).tobytes()


def test_reading_skips_payload(tile, graph):
    frame = bytearray(geozl.compress(tile, graph=graph, coeffs=VECTORS))
    frame[-1] ^= 0xFF
    assert geozl.coeffs(bytes(frame)) == VECTORS
