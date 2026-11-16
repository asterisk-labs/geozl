"""Library refusals raise geozl.GeoZLError with the ZL error code."""

import pickle
from pathlib import Path

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d, _error  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

# ZL_ErrorCode values from extern/openzl/include/openzl/zl_errors_types.h
_SRC_TOO_SMALL = 3
_PARAMETER_INVALID = 21
_GRAPH_INVALID = 30
_FORMAT_UNSUPPORTED = 60


def _tile(dtype=np.uint16):
    y, x = np.mgrid[0:32, 0:48]
    return (1000 + 3 * x + 2 * y).astype(dtype)


def _frame(tile=None):
    tile = _tile() if tile is None else tile
    return geozl.compress(tile, geozl.graph(tile, "planar>zigzag>pfor"))


def _invalid(exc):
    return isinstance(exc, ValueError)


def test_geozl_error_is_exported_as_a_runtime_error():
    assert "GeoZLError" in geozl.__all__
    assert issubclass(geozl.GeoZLError, RuntimeError)
    assert not issubclass(geozl.GeoZLError, ValueError)
    exc = geozl.GeoZLError("refused")
    assert str(exc) == "refused" and exc.code is None
    assert geozl.GeoZLError("refused", 7).code == 7


@pytest.mark.parametrize("code, invalid", [
    (_PARAMETER_INVALID, True), (_GRAPH_INVALID, True),
    (_SRC_TOO_SMALL, False), (_FORMAT_UNSUPPORTED, False),
])
def test_an_invalid_parameter_or_graph_is_also_a_value_error(code, invalid):
    exc = _error.refusal("refused", code)
    assert isinstance(exc, geozl.GeoZLError)
    assert exc.code == code
    assert _invalid(exc) is invalid


def test_an_unknown_method_is_an_invalid_request():
    with pytest.raises(geozl.GeoZLError) as caught:
        geozl.graph(_tile(), "planar>nothing")
    assert caught.value.code == _PARAMETER_INVALID
    assert _invalid(caught.value)
    assert str(caught.value).endswith(f"(ZL error code {_PARAMETER_INVALID})")


def test_a_method_the_sample_width_refuses_is_an_invalid_graph():
    with pytest.raises(ValueError, match="1-byte elements") as caught:
        geozl.graph(_tile(np.uint8), "id>transpose>entropy")
    assert isinstance(caught.value, geozl.GeoZLError)
    assert caught.value.code == _GRAPH_INVALID


def test_a_lossy_tile_outside_the_grid_is_an_invalid_request():
    tile = _tile(np.float32)
    g = geozl.graph(tile, "planar>zigzag>entropy", error=1)
    with pytest.raises(ValueError, match=r"^geozl.compress failed \(method=") as caught:
        geozl.compress(-tile, g)
    assert isinstance(caught.value, geozl.GeoZLError)
    assert caught.value.code == _PARAMETER_INVALID


def test_a_profile_where_every_recipe_fails_keeps_the_first_reason():
    with pytest.raises(geozl.GeoZLError, match="families are LINEAR") as caught:
        geozl.profile(_tile(), error="NOPE:1", reps=1)
    assert caught.value.code == _PARAMETER_INVALID
    assert _invalid(caught.value)


def test_a_corrupt_frame_is_a_refusal_but_not_an_invalid_request():
    with pytest.raises(geozl.GeoZLError, match="decompress failed") as caught:
        geozl.decompress(_frame()[:-3])
    assert caught.value.code == _SRC_TOO_SMALL
    assert not _invalid(caught.value)

    with pytest.raises(geozl.GeoZLError, match="unreadable frame") as caught:
        geozl.decompress(b"\x01\x02\x03\x04" * 4)
    assert isinstance(caught.value.code, int) and caught.value.code > 0


def test_a_frame_that_decodes_with_an_invalid_graph_code_is_not_a_bad_request():
    frame = _frame()
    for at in range(len(frame)):
        bad = frame[:at] + b"\x00" + frame[at + 1:]
        try:
            geozl.decompress(bad, verify=False)
        except geozl.GeoZLError as exc:
            if exc.code == _GRAPH_INVALID:
                assert type(exc) is geozl.GeoZLError
                break
    else:
        pytest.fail("no corruption decoded with ZL error code 30")


def test_an_unreadable_frame_has_a_code_in_coeffs():
    with pytest.raises(geozl.GeoZLError, match="unreadable frame") as caught:
        geozl.coeffs(b"\x01\x02\x03\x04" * 4)
    assert isinstance(caught.value.code, int) and caught.value.code > 0


def test_an_old_frame_has_the_unsupported_format_code():
    old = (Path(__file__).parent / "fixtures" / "openzl-format21.zl").read_bytes()
    with pytest.raises(geozl.GeoZLError, match="predates OpenZL format 22") as caught:
        geozl.coeffs(old)
    assert caught.value.code == _FORMAT_UNSUPPORTED


def test_a_frame_that_decodes_short_is_refused(monkeypatch):
    lib = _2d._load_lib_full()

    class Short:
        def __getattr__(self, name):
            return getattr(lib, name)

        def geozl_2d_decompress_c(self, *args):
            rc = lib.geozl_2d_decompress_c(*args)
            args[4][0] -= 2
            return rc

    frame = _frame()
    monkeypatch.setattr(_2d, "_load_lib_full", Short)
    with pytest.raises(geozl.GeoZLError, match="held 3070 bytes, not 3072") as caught:
        geozl.decompress(frame)
    assert caught.value.code is None
    assert not _invalid(caught.value)


def test_an_exceeded_limit_stays_an_argument_error():
    with pytest.raises(ValueError, match="above the 10 allowed") as caught:
        geozl.decompress(_frame(), max_output_size=10)
    assert not isinstance(caught.value, geozl.GeoZLError)


def test_a_refusal_survives_pickling_with_its_class_and_code():
    with pytest.raises(geozl.GeoZLError) as caught:
        geozl.graph(_tile(), "planar>nothing")
    back = pickle.loads(pickle.dumps(caught.value))
    assert type(back) is type(caught.value)
    assert back.code == _PARAMETER_INVALID and str(back) == str(caught.value)
