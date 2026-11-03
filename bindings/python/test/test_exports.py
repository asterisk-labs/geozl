"""libgeozl exports its own API and none of what it links."""

import ctypes
import ctypes.util
import os
from pathlib import Path

import pytest

geozl = pytest.importorskip("geozl")

from geozl import _ffi  # noqa: E402  after importorskip, on purpose

try:
    _ffi._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)


@pytest.fixture(scope="module")
def libgeozl():
    return ctypes.CDLL(os.environ.get("GEOZL_FULL_LIB")
                       or _ffi._bundled("geozl", reject="kernels")
                       or ctypes.util.find_library("geozl"))


def test_the_api_is_exported(libgeozl):
    assert hasattr(libgeozl, "geozl_2d_compress_c")


# An installed OpenZL exports its own symbols, and libgeozl links it as it is.
CORE = Path(__file__).resolve().parents[3] / "core"
_SYSTEM_OPENZL = any(
    cache.is_file() and "GEOZL_USE_SYSTEM_OPENZL:BOOL=ON" in cache.read_text()
    for cache in (CORE / "build" / "CMakeCache.txt",
                  CORE / "build-san" / "CMakeCache.txt"))


@pytest.mark.skipif(_SYSTEM_OPENZL, reason="built on an installed OpenZL")
@pytest.mark.parametrize("symbol", ["ZSTD_compress", "ZSTD_getErrorCode",
                                    "LZ4_compress_default", "LZ4F_compressFrame",
                                    "ZL_Compressor_create"])
def test_a_linked_library_stays_off_the_symbol_table(libgeozl, symbol):
    assert not hasattr(libgeozl, symbol)
