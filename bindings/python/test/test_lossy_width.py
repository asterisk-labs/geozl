"""Arguments the graph builder refuses before a scan or a node can misuse them."""

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl._dtype import dtype_code  # noqa: E402  after importorskip, on purpose
from geozl._ffi import _load_lib_full, _ptr, ffi  # noqa: E402

try:
    lib = _load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

METHOD = b"planar>zigzag>entropy"
PARAMETER_INVALID = 21  # ZL_ErrorCode_parameter_invalid
F64 = dtype_code(np.dtype(np.float64))
U16 = dtype_code(np.dtype(np.uint16))


def _open(src, *, elt, dtype, error=ffi.NULL, mode=0):
    out = ffi.new("geozl_2d_graph**")
    why = ffi.new("char[]", 256)
    rc = lib.geozl_2d_graph_open_c(out, METHOD, src.size // 2 or 1, 1, error, dtype,
                                   mode, 0, _ptr(src), src.size, elt, why, len(why))
    if rc == 0:
        lib.geozl_2d_graph_close_c(out[0])
    return rc, ffi.string(why).decode()


@pytest.mark.parametrize("recipe", [b"LINEAR:MAX_ERROR=1", b"LOG:MAX_ERROR=1%",
                                    b"SQRT:MAX_ERROR=1N"])
@pytest.mark.parametrize("elt", [1, 2, 4])
def test_a_lossy_scan_does_not_read_past_a_narrower_element(recipe, elt):
    # 64 elements of 1 byte read as float64 is 512 bytes out of a 64 byte buffer.
    src = np.zeros(64, dtype=np.uint8)
    rc, why = _open(src, elt=elt, dtype=F64, error=recipe)
    assert rc == PARAMETER_INVALID
    assert why == "dtype width is not eltWidth"


def test_a_lossy_width_that_matches_still_opens():
    src = np.arange(64, dtype=np.float64)
    rc, why = _open(src, elt=8, dtype=F64, error=b"LINEAR:MAX_ERROR=1")
    assert (rc, why) == (0, "")


def test_lossless_does_not_read_the_dtype_so_it_may_disagree():
    # _2d passes dtype 0 for a dtype with no code, here one that is 4 bytes wide.
    src = np.arange(64, dtype=np.int32)
    rc, why = _open(src, elt=4, dtype=0)
    assert (rc, why) == (0, "")


@pytest.mark.parametrize("mode", [-1, 3, 7])
def test_an_unknown_nodata_mode_is_refused_for_what_it_is(mode):
    src = np.arange(64, dtype=np.uint16)
    rc, why = _open(src, elt=2, dtype=U16, mode=mode)
    assert rc == PARAMETER_INVALID
    assert why == "unknown nodata mode"
