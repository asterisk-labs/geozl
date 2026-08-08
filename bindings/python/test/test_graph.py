"""The graph handle. It exists so a compressor survives between tiles, so what
matters is that it does, and that the frames stay independent of each other.
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

GRAPH = "planar>zigzag>transpose>entropy"
W = 32


def _tile(seed=0, shape=(W, W), dtype=np.int16):
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape)
    return ((x * 3 + y * 5) % 400 + rng.integers(0, 40, shape)).astype(dtype)


def _back(frame, like):
    return geozl.decompress(frame).view(like.dtype).reshape(like.shape)


def test_a_graph_carries_what_it_was_built_with():
    arr = _tile()
    g = geozl.graph(arr, GRAPH)
    assert (g.method, g.width, g.dtype, g.itemsize) == (GRAPH, W, arr.dtype, 2)
    assert g.error is None


def test_one_graph_compresses_many_tiles():
    tiles = [_tile(seed) for seed in range(8)]
    g = geozl.graph(tiles[0], GRAPH)
    for t in tiles:
        assert np.array_equal(_back(geozl.compress(t, graph=g), t), t)


def test_frames_from_one_graph_are_independent_of_each_other():
    """The reason the graph exists is random access, so a frame has to decode
    without the ones written before it. If the context carried state across
    compressions the second frame would only read after the first."""
    a, b = _tile(1), _tile(2)
    g = geozl.graph(a, GRAPH)
    first = geozl.compress(a, graph=g)
    second = geozl.compress(b, graph=g)
    assert first == geozl.compress(a, graph=g)
    assert np.array_equal(_back(second, b), b)
    assert np.array_equal(_back(first, a), a)


def test_a_shared_graph_writes_what_a_fresh_one_would():
    """Byte for byte, or the graph is doing something a one-shot compress is
    not and the two paths have drifted."""
    for seed in range(4):
        t = _tile(seed)
        shared = geozl.graph(_tile(0), GRAPH)
        assert geozl.compress(t, graph=shared) == \
               geozl.compress(t, graph=geozl.graph(t, GRAPH))


def test_a_shorter_tile_goes_through_the_same_graph():
    """Edge tiles get cut rather than padded, so the last tile of a row is
    shorter than the rest and still belongs to that graph."""
    full = _tile()
    edge = full[:, :W - 8].copy()
    g = geozl.graph(full, GRAPH)
    assert np.array_equal(_back(geozl.compress(edge, graph=g), edge), edge)


def test_compress_wants_a_graph_and_not_a_recipe():
    with pytest.raises(TypeError, match="geozl.Graph"):
        geozl.compress(_tile(), graph=GRAPH)


def test_compress_refuses_a_tile_of_another_element_width():
    g = geozl.graph(_tile(), GRAPH)
    with pytest.raises(ValueError, match="2-byte elements"):
        geozl.compress(_tile().astype(np.int32), graph=g)


def test_a_lossy_graph_quantizes_every_tile_on_one_grid():
    """The plan is cut once, against the raster graph() was given, so two tiles
    of one product reconstruct the same value the same way."""
    rng = np.random.default_rng(3)
    product = rng.uniform(10.0, 400.0, (64, 64)).astype(np.float32)
    a, b = product[:32].copy(), product[32:].copy()
    a[0, 0] = b[0, 0] = np.float32(123.5)
    g = geozl.graph(product, GRAPH, error="LINEAR:MAX_ERROR=2")
    assert _back(geozl.compress(a, graph=g), a)[0, 0] == \
           _back(geozl.compress(b, graph=g), b)[0, 0]
