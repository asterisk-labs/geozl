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
