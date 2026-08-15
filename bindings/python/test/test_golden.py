"""Decode frames written by GeoZL 0.13.0."""

import hashlib
import json
from pathlib import Path

import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

GOLDEN = Path(__file__).resolve().parent / "golden"
MANIFEST = json.loads((GOLDEN / "manifest.json").read_text())
FRAMES = MANIFEST["frames"]

def _ids(entry):
    return entry["name"]


def test_manifest_matches_the_fixtures():
    assert MANIFEST["written_by"] == "geozl 0.13.0"
    expected = {entry["file"] for entry in FRAMES}
    present = {path.name for path in (GOLDEN / "frames").glob("*.zl")}
    assert present == expected


@pytest.mark.parametrize("entry", FRAMES, ids=_ids)
def test_released_frame_still_decodes(entry):
    raw = (GOLDEN / "frames" / entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256_frame"]
    out = geozl.decompress(raw)
    assert hashlib.sha256(out.tobytes()).hexdigest() == entry["sha256_decoded"]
