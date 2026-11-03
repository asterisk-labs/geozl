"""The pre-commit hook checks the VERSION a commit would record."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[3] / ".githooks" / "pre-commit"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None or shutil.which("git") is None,
    reason="needs git and a POSIX bash")


def _git(repo, *args, stdin=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=repo, env=env, input=stdin,
                          capture_output=True, text=True, check=True).stdout


def _hook(repo):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["bash", str(HOOK)], cwd=repo, env=env,
                          capture_output=True, text=True).returncode


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    return tmp_path


def _stage(repo, version):
    (repo / "VERSION").write_text(version + "\n")
    _git(repo, "add", "VERSION")


@pytest.mark.parametrize("version", ["0.20.0", "0.20.0-alpha.1", "1.2.3-beta.2",
                                     "1.2.3-rc.10"])
def test_a_version_the_r_package_can_map_passes(repo, version):
    _stage(repo, version)
    assert _hook(repo) == 0


@pytest.mark.parametrize("version", ["", "1.2", "0.5.0-dev", "1.2.3-rc", "1.2.3-RC.1",
                                     "1.2.3-rc.1.1", "v1.2.3"])
def test_any_other_version_is_refused(repo, version):
    _stage(repo, version)
    assert _hook(repo) != 0


def test_the_index_decides_not_the_worktree(repo):
    _stage(repo, "not-a-version")
    (repo / "VERSION").write_text("1.2.3\n")
    assert _hook(repo) != 0

    _stage(repo, "1.2.3")
    (repo / "VERSION").write_text("not-a-version\n")
    assert _hook(repo) == 0


def test_an_unstaged_version_is_ignored(repo):
    (repo / "VERSION").write_text("not-a-version\n")
    assert _hook(repo) == 0


def test_a_long_staged_list_does_not_hide_the_version(repo):
    # More names than a pipe holds, sorted after VERSION, so a reader that stops
    # at the first match would kill the writer.
    blob = _git(repo, "hash-object", "-w", "--stdin", stdin="").strip()
    names = "".join(f"100644 {blob}\tz{i:06d}_{'x' * 80}\n" for i in range(4000))
    _git(repo, "update-index", "--index-info", stdin=names)
    _stage(repo, "not-a-version")
    assert _hook(repo) != 0
