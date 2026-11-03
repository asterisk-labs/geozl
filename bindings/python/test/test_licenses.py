"""Check the licence texts included in binary distributions."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LICENSES = ROOT / "licenses"
NOTICE = ROOT / "NOTICE"
COPYRIGHTS = ROOT / "bindings" / "r" / "inst" / "COPYRIGHTS"

VENDORED = {
    "LICENSE.OpenZL": "extern/openzl/LICENSE",
    "LICENSE.Zstandard": "extern/openzl/deps/zstd/LICENSE",
    "LICENSE.LZ4": "extern/openzl/deps/lz4/lib/LICENSE",
}

# Upstream keeps these only in a source file's comment, so the text here is that
# comment without its markers. It cannot match byte for byte, and the check below
# compares the words instead.
FROM_HEADER = {
    "LICENSE.xxHash": "extern/openzl/src/openzl/shared/xxhash.h",
    "LICENSE.pdqsort": "extern/openzl/src/openzl/shared/detail/pdqsort-inl.h",
    "LICENSE.divsufsort": "extern/openzl/deps/zstd/lib/dictBuilder/divsufsort.c",
}

# Upstream's text lives outside this repository, so only its presence is checked.
# JoinPaths.cmake names its licences by SPDX id, jtojnar/cmake-snips has the text.
# The other two come from Emscripten 4.0.19, the version the workflows install.
EXTERNAL = {"LICENSE.JoinPaths", "LICENSE.musl", "LICENSE.Emscripten"}

IN_LIBGEOZL = {"LICENSE.OpenZL", "LICENSE.xxHash", "LICENSE.pdqsort",
               "LICENSE.Zstandard", "LICENSE.LZ4"}
IN_R_SOURCE = IN_LIBGEOZL | {"LICENSE.divsufsort", "LICENSE.JoinPaths"}

EVERY_LICENSE = set(VENDORED) | set(FROM_HEADER) | EXTERNAL


def _words(text):
    """Text without comment markers, list bullets, line breaks or indentation."""
    return " ".join(re.sub(r"(?m)^[\s/*]+", "", text).split())


def test_the_repo_carries_every_notice():
    assert (ROOT / "LICENSE").is_file()
    assert NOTICE.is_file()
    names = {p.name for p in LICENSES.glob("LICENSE.*")}
    assert names == EVERY_LICENSE, f"licenses/ holds {names}"


@pytest.mark.parametrize("name", sorted(IN_R_SOURCE))
def test_r_package_carries_the_same_licence(name):
    r_copy = ROOT / "bindings" / "r" / "inst" / "licenses" / name
    assert r_copy.read_bytes() == (LICENSES / name).read_bytes()
    assert f"licenses/{name}" in COPYRIGHTS.read_text()


def test_r_package_carries_lz4_non_lib_licence():
    upstream = ROOT / "extern" / "openzl" / "deps" / "lz4" / "programs" / "COPYING"
    if not upstream.is_file():
        pytest.skip(f"{upstream} not checked out")
    r_copy = ROOT / "bindings" / "r" / "inst" / "licenses" / "COPYING.LZ4"
    assert r_copy.read_bytes() == upstream.read_bytes()
    assert "licenses/COPYING.LZ4" in COPYRIGHTS.read_text()


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_the_vendored_text_is_the_upstream_text(name):
    upstream = ROOT / VENDORED[name]
    if not upstream.is_file():
        pytest.skip(f"{upstream} not checked out")
    assert (LICENSES / name).read_bytes() == upstream.read_bytes(), (
        f"{name} drifted from {VENDORED[name]}; re-copy it and check whether "
        "the version named in NOTICE moved too"
    )


@pytest.mark.parametrize("name", sorted(FROM_HEADER))
def test_the_header_text_is_the_upstream_comment(name):
    upstream = ROOT / FROM_HEADER[name]
    if not upstream.is_file():
        pytest.skip(f"{upstream} not checked out")
    assert _words((LICENSES / name).read_text()) in _words(upstream.read_text()), (
        f"{name} drifted from the comment in {FROM_HEADER[name]}"
    )


def test_joinpaths_text_names_the_holder_of_the_file():
    upstream = (ROOT / "extern/openzl/deps/zstd/build/cmake/CMakeModules"
                / "JoinPaths.cmake")
    if not upstream.is_file():
        pytest.skip(f"{upstream} not checked out")
    assert "Jan Tojnar" in upstream.read_text()
    assert "Jan Tojnar" in (LICENSES / "LICENSE.JoinPaths").read_text()


@pytest.mark.parametrize("name", sorted(EVERY_LICENSE))
def test_notice_accounts_for_every_component(name):
    text = NOTICE.read_text()
    assert f"licenses/{name}" in text, f"NOTICE does not list {name}"


@pytest.mark.parametrize("name", sorted(IN_LIBGEOZL))
def test_the_wheel_build_and_its_check_list_the_same_texts(name):
    assert f"licenses/{name}" in (ROOT / "bindings/python/hatch_build.py").read_text()
    assert f'"{name}"' in (ROOT / "bindings/python/check_wheel.py").read_text()


@pytest.mark.parametrize("name", sorted(IN_R_SOURCE))
def test_the_r_vendor_script_copies_the_texts(name):
    assert f'"{name}"' in (ROOT / "tools/sync_r_vendor.py").read_text()
