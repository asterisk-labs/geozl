import os
import platform
import subprocess
import sys

import pytest

import geozl
from geozl import _2d

ORDER = ("scalar", "sse2", "avx2", "neon")

# FULL=OFF leaves libgeozl out, and with it geozl.compress.
try:
    _2d._load_lib_full()
    HAVE_FULL = True
except OSError:  # pragma: no cover - depends on how the build was configured
    HAVE_FULL = False

# A GEOZL_NO_SIMD build carries nothing but the scalar path, on purpose.
VECTORLESS = geozl.simd_info()["built"] == ["scalar"]


def run_with(cap, snippet):
    """The cap is read when the library loads, so it takes a fresh process.
    macOS strips the dyld preload from the child, so reinject the runtime path
    the way test_malformed does, or a sanitizer build aborts on load."""
    env = dict(os.environ)
    env.pop("GEOZL_SIMD", None) if cap is None else env.update(GEOZL_SIMD=cap)
    rt = env.get("GEOZL_ASAN_RT")
    if rt:
        var = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
        env[var] = rt
    out = subprocess.run([sys.executable, "-c", snippet], env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def both():
    """Paths this build carries that this processor can also run, before any
    ceiling."""
    info = geozl.simd_info()
    return [p for p in ORDER if p in info["built"] and p in info["cpu"]]


def under(cap):
    """The same, capped. GEOZL_SIMD is set in this process on the matrix rows
    that pin a path, so a test that reads active has to apply it too."""
    paths = both()
    if cap in ORDER:
        paths = [p for p in paths if ORDER.index(p) <= ORDER.index(cap)]
    return paths or ["scalar"]


def test_shape():
    info = geozl.simd_info()
    assert set(info) == {"built", "cpu", "active"}
    assert set(info["built"]) <= set(ORDER)
    assert set(info["cpu"]) <= set(ORDER)
    assert info["active"] in ORDER


def test_scalar_is_always_there():
    info = geozl.simd_info()
    assert "scalar" in info["built"] and "scalar" in info["cpu"]


def test_active_is_the_best_of_both():
    assert both()
    assert geozl.simd_info()["active"] == under(os.environ.get("GEOZL_SIMD", ""))[-1]


@pytest.mark.parametrize("cap", ORDER)
def test_env_caps_but_never_raises(cap):
    """The child gets its own ceiling, whatever this process was given."""
    got = run_with(cap, "import geozl; print(geozl.simd_info()['active'])")
    assert got == under(cap)[-1]


def test_unknown_env_value_is_ignored():
    """Against a child with no ceiling, not against this process, which the
    matrix may have pinned."""
    snippet = "import geozl; print(geozl.simd_info()['active'])"
    assert run_with("turbo", snippet) == run_with(None, snippet)


@pytest.mark.skipif(geozl.simd_info()["active"] == "scalar",
                    reason="no vector path to compare against")
@pytest.mark.skipif(not HAVE_FULL,
                    reason="libgeozl not built, rebuild with FULL=ON")
def test_vector_and_scalar_agree_byte_for_byte():
    """Compares the frame and not the array. The array survives a bad fill
    because decode overwrites everything under the mask, the frame does not."""
    snippet = (
        "import numpy as np, geozl, hashlib\n"
        "rng = np.random.default_rng(11)\n"
        "t = (1200 + rng.normal(0, 40, (256, 256))).round().astype(np.int16)\n"
        "t[40:90, 10:200] = -9999\n"
        "t[::7, ::11] = -9999\n"
        "g = geozl.graph(t, 'planar>zigzag>entropy', nodata=-9999)\n"
        "f = geozl.compress(t, graph=g)\n"
        "print(hashlib.sha256(f).hexdigest())\n"
    )
    assert run_with("scalar", snippet) == run_with(None, snippet)


@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="AVX2 is an x86 question")
@pytest.mark.skipif(VECTORLESS, reason="built with GEOZL_NO_SIMD")
def test_build_keeps_the_paths_the_cpu_offers():
    info = geozl.simd_info()
    missing = [p for p in info["cpu"] if p not in info["built"]]
    assert not missing, f"cpu runs {missing}, build does not carry it"
