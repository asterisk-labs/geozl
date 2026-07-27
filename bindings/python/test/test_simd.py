import os
import platform
import subprocess
import sys

import pytest

import geozl

ORDER = ("scalar", "sse2", "avx2", "neon")


def run_with(cap, snippet):
    """The cap is read when the library loads, so it takes a fresh process."""
    env = dict(os.environ)
    env.pop("GEOZL_SIMD", None) if cap is None else env.update(GEOZL_SIMD=cap)
    out = subprocess.run([sys.executable, "-c", snippet], env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def usable():
    info = geozl.simd_info()
    return [p for p in ORDER if p in info["built"] and p in info["cpu"]]


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
    assert usable()
    assert geozl.simd_info()["active"] == usable()[-1]


@pytest.mark.parametrize("cap", ORDER)
def test_env_caps_but_never_raises(cap):
    got = run_with(cap, "import geozl; print(geozl.simd_info()['active'])")
    under = [p for p in usable() if ORDER.index(p) <= ORDER.index(cap)]
    assert got == (under[-1] if under else "scalar")


def test_unknown_env_value_is_ignored():
    got = run_with("turbo", "import geozl; print(geozl.simd_info()['active'])")
    assert got == geozl.simd_info()["active"]


@pytest.mark.skipif(geozl.simd_info()["active"] == "scalar",
                    reason="no vector path to compare against")
def test_vector_and_scalar_agree_byte_for_byte():
    """Compares the frame and not the array. The array survives a bad fill
    because decode overwrites everything under the mask, the frame does not."""
    snippet = (
        "import numpy as np, geozl, hashlib\n"
        "rng = np.random.default_rng(11)\n"
        "t = (1200 + rng.normal(0, 40, (256, 256))).round().astype(np.int16)\n"
        "t[40:90, 10:200] = -9999\n"
        "t[::7, ::11] = -9999\n"
        "f = geozl.compress(t, method='planar>zigzag>entropy', nodata=-9999)\n"
        "print(hashlib.sha256(f).hexdigest())\n"
    )
    assert run_with("scalar", snippet) == run_with(None, snippet)


@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"),
                    reason="AVX2 is an x86 question")
def test_build_keeps_the_paths_the_cpu_offers():
    info = geozl.simd_info()
    missing = [p for p in info["cpu"] if p not in info["built"]]
    assert not missing, f"cpu runs {missing}, build does not carry it"
