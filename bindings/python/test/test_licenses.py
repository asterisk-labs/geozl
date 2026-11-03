"""Check the licence texts included in binary distributions."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LICENSES = ROOT / "licenses"
NOTICE = ROOT / "NOTICE"

VENDORED = {
    "LICENSE.OpenZL": "extern/openzl/LICENSE",
    "LICENSE.Zstandard": "extern/openzl/deps/zstd/LICENSE",
    "LICENSE.LZ4": "extern/openzl/deps/lz4/lib/LICENSE",
}
def test_the_repo_carries_every_notice():
    assert (ROOT / "LICENSE").is_file()
    assert NOTICE.is_file()
    names = {p.name for p in LICENSES.glob("LICENSE.*")}
    assert names == set(VENDORED), f"licenses/ holds {names}"


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_r_package_carries_the_same_licence(name):
    r_copy = ROOT / "bindings" / "r" / "inst" / "licenses" / name
    assert r_copy.read_bytes() == (LICENSES / name).read_bytes()


def test_r_package_carries_lz4_non_lib_licence():
    upstream = ROOT / "extern" / "openzl" / "deps" / "lz4" / "programs" / "COPYING"
    if not upstream.is_file():
        pytest.skip(f"{upstream} not checked out")
    r_copy = ROOT / "bindings" / "r" / "inst" / "licenses" / "COPYING.LZ4"
    assert r_copy.read_bytes() == upstream.read_bytes()
    assert "licenses/COPYING.LZ4" in (
        ROOT / "bindings" / "r" / "inst" / "COPYRIGHTS"
    ).read_text()


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_the_vendored_text_is_the_upstream_text(name):
    upstream = ROOT / VENDORED[name]
    if not upstream.is_file():
        pytest.skip(f"{upstream} not checked out")
    assert (LICENSES / name).read_bytes() == upstream.read_bytes(), (
        f"{name} drifted from {VENDORED[name]}; re-copy it and check whether "
        "the version named in NOTICE moved too"
    )


@pytest.mark.parametrize("name", sorted(VENDORED))
def test_notice_accounts_for_every_component(name):
    text = NOTICE.read_text()
    assert f"licenses/{name}" in text, f"NOTICE does not list {name}"
