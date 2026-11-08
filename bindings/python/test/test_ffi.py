import ctypes.util
import importlib
import importlib.metadata as md
import os
import subprocess
import sys

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _ffi  # noqa: E402  after importorskip, on purpose

try:
    _ffi._load_lib_full()
    _HAVE_FULL = True
except OSError:  # pragma: no cover - depends on how the build was configured
    _HAVE_FULL = False

needs_full = pytest.mark.skipif(not _HAVE_FULL,
                                reason="libgeozl not built, rebuild with FULL=ON")


def _blind(monkeypatch):
    """Hide every way of finding a library, so the lookup has to give up."""
    monkeypatch.delenv("GEOZL_LIB", raising=False)
    monkeypatch.delenv("GEOZL_FULL_LIB", raising=False)
    monkeypatch.setattr(_ffi, "_bundled_lib", lambda: None)
    monkeypatch.setattr(_ffi, "_bundled", lambda *a, **k: None)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)


def test_bundled_returns_none_when_nothing_matches():
    assert _ffi._bundled("a_library_that_is_not_shipped") is None


def test_the_kernels_lib_is_findable():
    # it is loaded already, so one of the two has to have worked
    assert _ffi._bundled_lib() or _ffi.os.environ.get("GEOZL_LIB")


@needs_full
def test_the_full_lib_lookup_skips_the_kernels_one():
    # "geozl" is a prefix of "geozl_kernels", so without the reject the full
    # lookup would load the wrong library
    assert _ffi._bundled("geozl", reject="kernels") != _ffi._bundled_lib()


def test_load_lib_names_the_variable_to_set(monkeypatch):
    _blind(monkeypatch)
    with pytest.raises(OSError, match="GEOZL_LIB"):
        _ffi._load_lib()


def test_load_lib_full_names_the_variable_to_set(monkeypatch):
    _blind(monkeypatch)
    monkeypatch.setattr(_ffi, "_lib_full", None)  # drop the cached handle
    with pytest.raises(OSError, match="GEOZL_FULL_LIB"):
        _ffi._load_lib_full()


def test_load_lib_falls_back_to_the_system_loader(monkeypatch):
    asked = []
    monkeypatch.delenv("GEOZL_LIB", raising=False)
    monkeypatch.setattr(_ffi, "_bundled_lib", lambda: None)
    monkeypatch.setattr(ctypes.util, "find_library", lambda n: asked.append(n))
    with pytest.raises(OSError):
        _ffi._load_lib()
    assert asked == ["geozl_kernels"]


def test_the_env_var_wins_over_the_bundled_lib(monkeypatch):
    monkeypatch.setenv("GEOZL_LIB", "/definitely/not/here.so")
    monkeypatch.setattr(_ffi, "_bundled_lib", lambda: "/ignored.so")
    # dlopen fails, but only after the env var was the one picked
    with pytest.raises(Exception) as e:
        _ffi._load_lib()
    assert "/definitely/not/here.so" in str(e.value)


@needs_full
def test_the_full_lib_handle_is_cached():
    assert _ffi._load_lib_full() is _ffi._load_lib_full()


def test_ptr_hands_back_something_the_kernels_can_write_through():
    assert _ffi._ptr(np.arange(8, dtype=np.uint8)) != _ffi.ffi.NULL


def test_version_is_reported():
    assert isinstance(geozl.__version__, str) and geozl.__version__


def test_version_falls_back_when_the_package_is_not_installed(monkeypatch):
    def missing(_name):
        raise md.PackageNotFoundError("geozl")

    monkeypatch.setattr(md, "version", missing)
    monkeypatch.delitem(sys.modules, "geozl", raising=False)
    try:
        assert importlib.import_module("geozl").__version__ == "0+unknown"
    finally:
        # leave the real module in place for whatever runs next
        monkeypatch.undo()
        del sys.modules["geozl"]
        importlib.import_module("geozl")


def test_all_promises_only_what_the_package_has():
    for name in geozl.__all__:
        assert hasattr(geozl, name), f"__all__ promises {name}, it is not there"

def test_register_decoders_names_openzl_when_it_is_missing():
    """Without openzl the package still imports and the low-level API says why
    it is unavailable, instead of failing on a name that was never bound."""
    script = (
        "import sys\n"
        "sys.modules['openzl'] = None\n"
        "import geozl\n"
        "assert 'lossless' not in geozl.__all__\n"
        "try:\n"
        "    geozl.register_decoders(object())\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.name == 'openzl', exc\n"
        "else:\n"
        "    raise SystemExit('register_decoders did not refuse')\n"
    )
    # macOS strips the dyld preload from the child, so reinject the runtime
    # path the way test_malformed does, or a sanitizer build aborts on load.
    env = dict(os.environ)
    rt = env.get("GEOZL_ASAN_RT")
    if rt:
        var = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
        env[var] = rt
    run = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env)
    assert run.returncode == 0, run.stderr
