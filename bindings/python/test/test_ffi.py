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
    subpackages = [getattr(geozl, name) for name in ("lossless", "lossy")
                   if name in geozl.__all__]
    for module in (geozl, *subpackages):
        for name in module.__all__:
            assert hasattr(module, name), \
                f"{module.__name__}.__all__ promises {name}, it is not there"


def _python(body, *, openzl):
    """Run body in a fresh interpreter, with openzl importable or not, and
    return what it prints."""
    script = ("" if openzl else "import sys\nsys.modules['openzl'] = None\n") + body
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
    return run.stdout


def test_register_decoders_names_openzl_when_it_is_missing():
    """Without openzl the package still imports and the low-level API says why
    it is unavailable, instead of failing on a name that was never bound."""
    _python(
        "import geozl\n"
        "assert 'lossless' not in geozl.__all__\n"
        "try:\n"
        "    geozl.register_decoders(object())\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert exc.name == 'openzl', exc\n"
        "else:\n"
        "    raise SystemExit('register_decoders did not refuse')\n",
        openzl=False)


def test_lossy_nodes_name_openzl_when_it_is_missing():
    """A node imported by name says what is missing, as its module would."""
    _python(
        "import geozl.lossy\n"
        "for name in ('QuantLinear', 'QuantLogDecoder', 'QuantSqrt'):\n"
        "    try:\n"
        "        exec(f'from geozl.lossy import {name}')\n"
        "    except ModuleNotFoundError as exc:\n"
        "        assert exc.name == 'openzl', exc\n"
        "    else:\n"
        "        raise SystemExit(name + ' imported')\n"
        "try:\n"
        "    geozl.lossy.Nothing\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('an unknown name resolved')\n",
        openzl=False)


def test_hints_resolve_at_run_time():
    """Runtime checkers read the hints, so every name in them must exist."""
    import typing

    import geozl
    for fn in (geozl.decompress, geozl.coeffs, geozl.compress, geozl.graph):
        typing.get_type_hints(fn)


def test_all_promises_only_what_the_package_has_without_openzl():
    """geozl.lossy stays, down to the noise fit, and the nodes leave __all__."""
    _python(
        "import geozl\n"
        "assert geozl.lossy.__all__ == ['Noise', 'fit_noise'], geozl.lossy.__all__\n"
        "for module in (geozl, geozl.lossy):\n"
        "    for name in module.__all__:\n"
        "        assert hasattr(module, name), (module.__name__, name)\n",
        openzl=False)


_SCENE = (
    "import numpy as np\n"
    "import geozl\n"
    "from geozl.lossy import fit_noise\n"
    "rng = np.random.default_rng(0)\n"
    "mu = np.sort(rng.uniform(50, 4000, 256 * 256)).reshape(256, 256)\n"
    "scene = (mu + rng.normal(0, np.sqrt(100 + mu))).astype(np.float32)\n"
    "recipe = fit_noise(scene).recipe(1)\n"
    "g = geozl.graph(scene, 'planar>zigzag>pfor', width=64, planes=1,\n"
    "                error=recipe)\n"
    "frame = geozl.compress(np.ascontiguousarray(scene[:64, :64]), g)\n"
    "print(recipe, frame.hex())\n"
)


@needs_full
def test_the_noise_fit_shares_a_curve_across_tiles_without_openzl():
    """A tiled SQRT graph needs A and B from fit_noise, and fit_noise needs
    only the kernels, so it has to work where OpenZL has no wheel, and give
    the curve and frame it gives with OpenZL."""
    out = _python(_SCENE, openzl=False)
    assert out.startswith("SQRT:MAX_ERROR=1N,A=")
    assert out == _python(_SCENE, openzl=True)
