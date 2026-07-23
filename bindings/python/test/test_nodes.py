import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

# Each of these owns a copy of the successor branch: four come out of the
# spatial_predictor factory, the other two spell it out themselves.
_NODES = [
    ("planar", lambda: geozl.lossless.Planar(8)),
    ("delta_w", lambda: geozl.lossless.DeltaW(8)),
    ("med", lambda: geozl.lossless.Med(8)),
    ("wp_static", lambda: geozl.lossless.WpStatic(8)),
    ("deinterleave", lambda: geozl.lossless.Deinterleave()),
    ("quant_linear", lambda: geozl.lossy.QuantLinear(4, np.uint16)),
]
_MAKERS = [m for _, m in _NODES]
_IDS = [n for n, _ in _NODES]


def _frame(node, arr):
    c = zl.Compressor()
    c.select_starting_graph(node(c, zl.graphs.Compress()(c)))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_accepts_an_unbound_successor(make):
    # zl.graphs.Compress() has no compressor yet, so the node has to
    # parameterize it instead of assuming a GraphID
    c = zl.Compressor()
    assert isinstance(make()(c, zl.graphs.Compress()), zl.GraphID)


@pytest.mark.parametrize("make", _MAKERS, ids=_IDS)
def test_accepts_a_bound_successor(make):
    c = zl.Compressor()
    assert isinstance(make()(c, zl.graphs.Compress()(c)), zl.GraphID)


@pytest.mark.parametrize("width", [0, 9999],
                         ids=["zero means one row", "wider than the tile"])
def test_a_width_that_cannot_tile_folds_to_one_row(width):
    # both fold to "one row of n" rather than raising, and the round trip still
    # has to come back exact
    arr = np.arange(32, dtype=np.uint16)
    d = zl.DCtx()
    geozl.register_decoders(d)
    frame = _frame(geozl.lossless.Planar(width), arr)
    out = d.decompress(frame)[0].content.as_nparray()
    assert np.array_equal(out.view(arr.dtype), arr)


def test_an_empty_tile_skips_the_width_check():
    # n == 0 short-circuits the guard, so a width that could never tile it is
    # still accepted instead of raising
    assert _frame(geozl.lossless.Planar(7), np.zeros(0, np.uint16)) is not None


@pytest.mark.parametrize("dtype,component", [
    (np.complex64, np.float32),
    (np.complex128, np.float64),
])
def test_component_dtype_gives_the_real_component(dtype, component):
    assert geozl.lossless.component_dtype(dtype) == np.dtype(component)


def test_component_dtype_refuses_anything_but_complex():
    with pytest.raises(KeyError):
        geozl.lossless.component_dtype(np.uint16)


def test_each_family_registers_only_its_own_decoders():
    lossless, lossy = [], []

    class Spy(zl.DCtx):
        def __init__(self, sink):
            super().__init__()
            self._sink = sink

        def register_custom_decoder(self, dec):
            self._sink.append(type(dec).__name__)
            return super().register_custom_decoder(dec)

    geozl.lossless.register_decoders(Spy(lossless))
    geozl.lossy.register_decoders(Spy(lossy))
    assert len(lossless) == 7
    assert len(lossy) == 1
    assert not set(lossless) & set(lossy)


def test_register_decoders_covers_both_families():
    seen = []

    class Spy(zl.DCtx):
        def register_custom_decoder(self, dec):
            seen.append(type(dec).__name__)
            return super().register_custom_decoder(dec)

    geozl.register_decoders(Spy())
    assert len(seen) == 8  # seven lossless plus one lossy