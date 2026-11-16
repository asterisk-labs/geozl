"""The argument rules the Python, R and Julia bindings share."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

METHOD = "planar>zigzag>pfor"


def _tile(dtype=np.uint16):
    y, x = np.mgrid[0:32, 0:48]
    return (1000 + 3 * x + 2 * y).astype(dtype)


def test_compress_takes_the_graph_by_position_or_by_name():
    tile = _tile()
    g = geozl.graph(tile, METHOD)
    frame = geozl.compress(tile, g)
    assert frame == geozl.compress(tile, graph=g)
    assert geozl.coeffs(geozl.compress(tile, g, coeffs=[[1]])) == ((1,),)
    with pytest.raises(TypeError):
        geozl.compress(tile, g, [[1]])
    with pytest.raises(TypeError, match="must be a geozl.Graph"):
        geozl.compress(tile, "planar>zigzag>pfor")


def test_a_graph_prints_its_fields_in_order():
    tile = _tile()
    assert repr(geozl.graph(tile, METHOD)) == (
        "<geozl.Graph planar>zigzag>pfor, uint16, row width 48, 1 plane, lossless>")
    cube = np.stack([tile, tile + 1]).astype(np.float32)
    assert repr(geozl.graph(cube, METHOD, error=2.0)) == (
        "<geozl.Graph planar>zigzag>pfor, float32, row width 48, 2 planes, "
        "LINEAR:MAX_ERROR=2.0>")


def test_a_graph_keeps_its_fields():
    g = geozl.graph(_tile(), METHOD, width=24, planes=2, error="1%", nodata=0)
    assert isinstance(g.dtype, np.dtype) and g.dtype == "uint16"
    assert (g.method, g.width, g.planes, g.itemsize, g.error, g.nodata) == (
        METHOD, 24, 2, 2, "LOG:MAX_ERROR=1%", 0)


@pytest.mark.parametrize("value", [8, np.int64(8), np.uint32(8), 2**32 - 1])
def test_width_takes_a_whole_number_up_to_uint32(value):
    assert geozl.graph(_tile(), METHOD, width=value).width == value


@pytest.mark.parametrize("name", ["width", "planes"])
@pytest.mark.parametrize("value", [True, np.True_, 3.7, 2.0, "48", [1]])
def test_geometry_refuses_what_is_not_a_whole_number(name, value):
    with pytest.raises(TypeError, match=f"^{name} must be a whole number$"):
        geozl.graph(_tile(), METHOD, **{name: value})
    with pytest.raises(TypeError, match=f"^{name} must be a whole number$"):
        geozl.profile(_tile(), reps=1, **{name: value})


@pytest.mark.parametrize("name", ["width", "planes"])
@pytest.mark.parametrize("value", [0, -1, 2**32, np.int64(-48)])
def test_geometry_refuses_a_count_outside_uint32(name, value):
    with pytest.raises(ValueError, match=f"^{name} must be from 1 to 4294967295$"):
        geozl.graph(_tile(), METHOD, **{name: value})


def test_an_explicit_width_of_zero_is_refused():
    tile = _tile()
    with pytest.raises(ValueError, match="from 1"):
        geozl.graph(tile, METHOD, width=0)
    # a raster with an empty last axis still defaults to it
    empty = np.zeros((8, 0), np.uint16)
    assert geozl.graph(empty, METHOD).width == 0


@pytest.mark.parametrize("method", ["planar>zigzag>pfor\x00", "planar\x00>zigzag>pfor"])
def test_a_nul_in_the_method_is_refused_and_not_cut(method):
    with pytest.raises(ValueError, match="NUL"):
        geozl.graph(_tile(), method)


def test_a_nul_in_an_error_recipe_or_prior_is_refused():
    with pytest.raises(ValueError, match="NUL"):
        geozl.graph(_tile(), METHOD, error="LINEAR:MAX_ERROR=2\x00junk")
    with pytest.raises(ValueError, match="is not one of"):
        geozl.profile(_tile(), prior="planar\x00", reps=1)


@pytest.mark.parametrize("value", ["5", b"5", True, np.False_, 1 + 2j, [5],
                                   np.array([5]), np.array(True)])
def test_nodata_must_be_one_real_number(value):
    with pytest.raises(TypeError, match="^nodata must be a number$"):
        geozl.graph(_tile(), METHOD, nodata=value)
    with pytest.raises(TypeError, match="^nodata must be a number$"):
        geozl.profile(_tile(), reps=1, nodata=value)


@pytest.mark.parametrize("value", [5, np.uint8(5), 5.0, np.float32(5), np.array(5)])
def test_nodata_takes_ints_floats_and_numpy_scalars(value):
    assert geozl.graph(_tile(), METHOD, nodata=value).nodata is value


@pytest.mark.parametrize("value, dtype", [
    (300, np.uint8), (-1, np.uint64), (2**64, np.uint64), (np.int64(-1), np.uint16),
    (float("inf"), np.int32), (1e300, np.float32), (10**400, np.float64),
])
def test_a_sentinel_outside_the_dtype_is_a_value_error(value, dtype):
    with pytest.raises(ValueError, match="^nodata does not fit the dtype$") as caught:
        geozl.graph(_tile(dtype), METHOD, nodata=value)
    assert not isinstance(caught.value, ArithmeticError)


@pytest.mark.parametrize("value", [
    Fraction(1, 2), Decimal("0.5"), Decimal("-2.25"), 2.5, np.float32(0.5),
    np.float64(-1.5), np.longdouble(0.5), np.array(0.5),
])
def test_a_fractional_sentinel_of_any_type_is_refused_on_integers(value):
    with pytest.raises(ValueError, match="is not a whole number"):
        geozl.graph(_tile(), METHOD, nodata=value)


@pytest.mark.parametrize("value", [Fraction(6, 2), Decimal("3"), Decimal("3.0"), 3.0])
def test_a_whole_sentinel_of_any_type_writes_the_same_frame(value):
    tile = _tile()
    tile[0, :5] = 3
    expected = geozl.compress(tile, geozl.graph(tile, METHOD, nodata=3))
    assert geozl.compress(tile, geozl.graph(tile, METHOD, nodata=value)) == expected


@pytest.mark.parametrize("value, dtype", [
    (Decimal("Infinity"), np.int32), (Decimal("1e400"), np.float64),
    (Fraction(10**400), np.float64), (Fraction(70000), np.uint16),
])
def test_a_fraction_or_decimal_outside_the_dtype_is_refused(value, dtype):
    with pytest.raises(ValueError, match="^nodata does not fit the dtype$"):
        geozl.graph(_tile(dtype), METHOD, nodata=value)


@pytest.mark.parametrize("nan", ["NaN", "-NaN", "sNaN", "-sNaN"])
def test_a_decimal_nan_is_nan(nan):
    with pytest.raises(ValueError, match="^a uint16 raster cannot use NaN as nodata$"):
        geozl.graph(_tile(), METHOD, nodata=Decimal(nan))
    tile = _tile(np.float32)
    tile[3, 4] = np.nan
    assert (geozl.compress(tile, geozl.graph(tile, METHOD, nodata=Decimal(nan))) ==
            geozl.compress(tile, geozl.graph(tile, METHOD, nodata=float("nan"))))
    from geozl._dtype import nodata_bits
    assert (nodata_bits(Decimal(nan), np.float32) ==
            nodata_bits(Decimal(nan.replace("s", "")), np.float32))


@pytest.mark.parametrize("dtype", [np.uint16, np.int8, np.uint64])
@pytest.mark.parametrize("nan", [float("nan"), np.float32("nan"), np.array(np.nan)])
def test_nan_cannot_be_the_sentinel_of_an_integer_raster(dtype, nan):
    name = np.dtype(dtype).name
    with pytest.raises(ValueError, match=f"^a {name} raster cannot use NaN as nodata$"):
        geozl.graph(_tile(dtype), METHOD, nodata=nan)


def test_nan_is_the_sentinel_of_a_float_raster():
    tile = _tile(np.float32)
    tile[3, 4] = np.nan
    g = geozl.graph(tile, "id>zstd", nodata=float("nan"))
    back = geozl.decompress(geozl.compress(tile, g)).view(np.float32)
    assert np.isnan(back[3 * 48 + 4])


@pytest.mark.parametrize("value", [1, 0, None, "no", 1.0])
def test_verify_must_be_a_boolean(value):
    frame = geozl.compress(_tile(), geozl.graph(_tile(), METHOD))
    with pytest.raises(TypeError, match="^verify must be a boolean$"):
        geozl.decompress(frame, verify=value)
    with pytest.raises(TypeError, match="^verify must be a boolean$"):
        geozl.profile(_tile(), reps=1, verify=value)


@pytest.mark.parametrize("value", [True, False, np.True_, np.False_])
def test_verify_takes_python_and_numpy_booleans(value):
    frame = geozl.compress(_tile(), geozl.graph(_tile(), METHOD))
    assert geozl.decompress(frame, verify=value).size == _tile().nbytes
    assert geozl.profile(_tile(), reps=1, verify=value)


@pytest.mark.parametrize("value", [1.5, 1.0, "5", True, None])
def test_reps_must_be_a_whole_number(value):
    with pytest.raises(TypeError, match="^reps must be a whole number$"):
        geozl.profile(_tile(), reps=value)


@pytest.mark.parametrize("value", [0, -3])
def test_reps_must_be_at_least_one(value):
    with pytest.raises(ValueError, match="at least 1"):
        geozl.profile(_tile(), reps=value)


@pytest.mark.parametrize("value", [2**31, 2**64, np.uint64(2**63)])
def test_reps_stops_at_int32(value):
    with pytest.raises(ValueError, match="^reps must be at most 2147483647, got "):
        geozl.profile(_tile(), reps=value)


_PADDED = [
    np.dtype({"names": ["a"], "formats": ["u1"], "offsets": [0], "itemsize": 4}),
    np.dtype([("a", "u1"), ("b", "<u2")], align=True),
    np.dtype([("a", np.dtype({"names": ["x"], "formats": ["u1"], "offsets": [0],
                              "itemsize": 2}))]),
    np.dtype([("a", np.dtype({"names": ["x"], "formats": ["u1"], "offsets": [0],
                              "itemsize": 2}), (2,))]),
]


@pytest.mark.parametrize("dtype", _PADDED, ids=["tail", "aligned", "nested", "subarray"])
def test_a_structured_dtype_with_padding_is_refused(dtype):
    tile = np.zeros((8, 8), dtype)
    with pytest.raises(ValueError, match="^a structured dtype must have no padding bytes$"):
        geozl.graph(tile, "id>zstd")
    with pytest.raises(ValueError, match="no padding bytes"):
        geozl.profile(tile, reps=1)


@pytest.mark.parametrize("dtype", [
    np.dtype([("a", "u1"), ("b", "<u2"), ("c", "u1")]),
    np.dtype([("a", "u1", (4,))]),
    np.dtype([("a", [("x", "u1"), ("y", "u1")]), ("b", "<u2")]),
    np.dtype({"names": ["a", "b"], "formats": ["<u4", "<u2"], "offsets": [0, 0],
              "itemsize": 4}),
    np.dtype("V4"), np.dtype("bool"), np.dtype("complex64"), np.dtype("datetime64[s]"),
], ids=["packed", "subarray", "nested", "overlap", "void", "bool", "complex", "datetime"])
def test_dtypes_without_padding_still_round_trip(dtype):
    tile = np.frombuffer(np.arange(64 * dtype.itemsize, dtype=np.uint8).tobytes(), dtype)
    frame = geozl.compress(tile, geozl.graph(tile, "id>zstd", width=8))
    assert geozl.decompress(frame).tobytes() == tile.tobytes()


def test_reps_takes_a_numpy_integer():
    assert geozl.profile(_tile(), reps=np.int64(1)).reps == 1
