import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_version_is_recorded_in_the_changelog():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{version}] - " in changelog
    assert f"[{version}]: https://github.com/asterisk-labs/geozl/compare/" in changelog
    assert (
        f"[Unreleased]: https://github.com/asterisk-labs/geozl/compare/"
        f"v{version}...HEAD"
    ) in changelog


def test_binding_versions_match_version():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    r = (ROOT / "bindings/r/DESCRIPTION").read_text(encoding="utf-8")
    julia = (ROOT / "bindings/julia/Project.toml").read_text(encoding="utf-8")

    assert re.search(r"(?m)^Version: (.*)$", r).group(1) == version
    assert re.search(r'(?m)^version = "(.*)"$', julia).group(1) == version


def test_r_check_status_needs_a_complete_known_result(tmp_path):
    script = ROOT / "tools/r_check_status.sh"
    cases = (
        ("* checking package dependencies ... OK\n", False),
        ("Status: OK\n", True),
        ("* checking compiled code ... WARNING\nObjects: lib/libopenzl.a\n"
         "Status: 1 WARNING\n", True),
        ("* checking examples ... WARNING\nStatus: 1 WARNING\n", False),
        ("Status: OK\nStatus: OK\n", False),
    )
    for i, (text, passes) in enumerate(cases):
        log = tmp_path / f"check-{i}.log"
        log.write_text(text, encoding="utf-8")
        result = subprocess.run(("sh", script, log), capture_output=True, check=False)
        assert (result.returncode == 0) is passes
