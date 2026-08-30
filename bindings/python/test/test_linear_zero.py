"""Regression tests for LINEAR graph construction on zero rasters."""

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on build configuration
    pytest.skip("libgeozl not built, rebuild with FULL=ON", allow_module_level=True)


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("nodata", [None, 0])
def test_linear_graph_accepts_an_all_zero_raster(dtype, nodata):
    raster = np.zeros((32, 32), dtype=dtype)
    graph = geozl.graph(
        raster,
        "planar>zigzag>pfor",
        error=10,
        nodata=nodata,
    )
    frame = geozl.compress(raster, graph=graph)
    decoded = geozl.decompress(frame).view(raster.dtype).reshape(raster.shape)
    assert np.array_equal(decoded, raster)


def test_linear_graph_still_rejects_a_raster_without_finite_samples():
    raster = np.full((32, 32), np.nan, dtype=np.float32)
    with pytest.raises(RuntimeError, match="no finite sample"):
        geozl.graph(raster, "planar>zigzag>pfor", error=10)
