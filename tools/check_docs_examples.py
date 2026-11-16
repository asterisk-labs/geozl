#!/usr/bin/env python3
"""Run the executable Python, R and Julia examples in docs/*.html.

Blocks on one page share a script. The checker verifies tabs and package
versions before running them; signature and data-run="skip" blocks are omitted.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

LANGUAGE = re.compile(r"\blanguage-(\w+)\b")
EXTENSIONS = {"python": "py", "r": "R", "julia": "jl"}
TAIL = 40


@dataclass
class Script:
    page: str
    language: str
    blocks: list[str]
    skipped: int

    @property
    def label(self) -> str:
        return f"{self.page} {self.language}"


@dataclass
class Result:
    script: Script
    ok: bool
    seconds: float
    output: str
    workdir: Path


class Examples(HTMLParser):
    def __init__(self, page: Path):
        super().__init__(convert_charrefs=True)
        self.page = page
        self.blocks: list[tuple[str, str, bool]] = []
        self.errors: list[str] = []
        self.tab_depth = 0
        self.tabs: list[str] = []
        self.pre: dict[str, str | None] | None = None
        self.code: dict[str, str | None] | None = None
        self.parts: list[str] = []
        self.code_language: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "div":
            if self.tab_depth:
                self.tab_depth += 1
            elif "code-tabs" in (values.get("class") or "").split():
                self.tab_depth, self.tabs = 1, []
        elif tag == "pre":
            self.pre = values
            self.code_language = None
            if self.tab_depth:
                self.tabs.append(values.get("data-lang") or "")
        elif tag == "code" and self.pre is not None:
            self.code = values
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.code is not None:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "code" and self.code is not None and self.pre is not None:
            classes = (self.code.get("class") or "").split()
            named = LANGUAGE.search(" ".join(classes))
            self.code_language = named[1] if named else None
            if self.code_language in EXTENSIONS:
                skip = "sig" in classes or self.pre.get("data-run") == "skip"
                self.blocks.append((self.code_language, "".join(self.parts), skip))
            self.code = None
        elif tag == "pre" and self.pre is not None:
            expected = (self.pre.get("data-lang") or "").lower()
            if expected in EXTENSIONS and self.code_language != expected:
                if not (expected == "python" and self.code_language == "sh"):
                    self.errors.append(
                        f"{self.page.name}: {expected} tab has no matching code block"
                    )
            self.pre = None
        elif tag == "div" and self.tab_depth:
            self.tab_depth -= 1
            if not self.tab_depth and sorted(self.tabs) != ["Julia", "Python", "R"]:
                self.errors.append(
                    f"{self.page.name}: code tabs must contain Python, R and Julia once each"
                )


def scripts(pages: list[Path], languages: list[str]) -> list[Script]:
    found = []
    for page in pages:
        parser = Examples(page)
        parser.feed(page.read_text())
        parser.close()
        if parser.tab_depth:
            parser.errors.append(f"{page.name}: unclosed code tabs")
        if parser.errors:
            raise ValueError("\n".join(parser.errors))
        for language in languages:
            blocks, skipped = [], 0
            for found_language, code, skip in parser.blocks:
                if found_language != language:
                    continue
                if skip:
                    skipped += 1
                    continue
                blocks.append(code)
            if blocks:
                found.append(Script(page.name, language, blocks, skipped))
    return found


def command(language: str, path: Path, julia_project: str | None) -> list[str]:
    if language == "python":
        return [sys.executable, str(path)]
    if language == "r":
        return ["Rscript", str(path)]
    project = [f"--project={julia_project}"] if julia_project else []
    return ["julia", "--startup-file=no", *project, str(path)]


def check_versions(languages: set[str], julia_project: str | None, env: dict[str, str]) -> None:
    expected = (ROOT / "VERSION").read_text().strip()
    commands = {
        "python": [sys.executable, "-c", "import geozl; print(geozl.__version__)"],
        "r": ["Rscript", "-e", 'cat(as.character(packageVersion("geozl")))'],
        "julia": [
            "julia", "--startup-file=no",
            *([f"--project={julia_project}"] if julia_project else []),
            "-e", "using GeoZL; print(Base.pkgversion(GeoZL))",
        ],
    }
    for language in sorted(languages):
        try:
            result = subprocess.run(commands[language], env=env, capture_output=True,
                                    text=True, timeout=90, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"cannot load GeoZL in {language}: {exc}") from exc
        if result.returncode != 0:
            raise ValueError(f"cannot load GeoZL in {language}: {result.stderr.strip()}")
        if result.stdout.strip() != expected:
            raise ValueError(f"{language} loads GeoZL {result.stdout.strip()}, expected {expected}")


def run(script: Script, args: argparse.Namespace, env: dict[str, str]) -> Result:
    workdir = Path(tempfile.mkdtemp(prefix=f"geozl-{Path(script.page).stem}-{script.language}-"))
    path = workdir / f"example.{EXTENSIONS[script.language]}"
    # Python, R and Julia all comment with #, so each block is marked for the
    # line numbers in a traceback.
    path.write_text("".join(f"# block {i + 1}\n{code}\n\n"
                            for i, code in enumerate(script.blocks)))
    started = time.monotonic()
    try:
        done = subprocess.run(command(script.language, path, args.julia_project),
                              cwd=workdir, env=env, capture_output=True, text=True, check=False,
                              timeout=args.timeout)
        ok, output = done.returncode == 0, done.stdout + done.stderr
    except subprocess.TimeoutExpired as exc:
        ok, output = False, f"timed out after {exc.timeout} s"
    except FileNotFoundError as exc:
        ok, output = False, f"{exc.filename} is not on PATH"
    seconds = time.monotonic() - started
    if ok and not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return Result(script, ok, seconds, output, workdir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", type=Path,
                        help="pages to run, every page under docs/ by default")
    parser.add_argument("--lang", action="append", choices=sorted(EXTENSIONS),
                        help="run only this language, may repeat")
    parser.add_argument("--julia-project",
                        default=os.environ.get("JULIA_PROJECT", str(ROOT / "bindings" / "julia")),
                        help="Julia environment with GeoZL")
    parser.add_argument("--jobs", type=int, default=3, help="scripts run at once")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per script")
    parser.add_argument("--keep", action="store_true", help="keep the work directories")
    parser.add_argument("--list", action="store_true", help="list the scripts, run nothing")
    args = parser.parse_args()

    pages = [page.resolve() for page in args.pages] or sorted(DOCS.glob("*.html"))
    try:
        todo = scripts(pages, args.lang or sorted(EXTENSIONS))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.list:
        for script in todo:
            print(f"{script.label}: {len(script.blocks)} blocks, {script.skipped} skipped")
        return 0
    if not todo:
        print("no examples to run")
        return 1

    env = dict(os.environ)
    try:
        check_versions({script.language for script in todo}, args.julia_project, env)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda script: run(script, args, env), todo))

    failed = [result for result in results if not result.ok]
    for result in results:
        script = result.script
        print(f"{'PASS' if result.ok else 'FAIL'} {script.label}: {len(script.blocks)} blocks, "
              f"{script.skipped} skipped, {result.seconds:.0f} s")
    for result in failed:
        print(f"\n--- {result.script.label}, kept in {result.workdir}")
        print("\n".join(result.output.rstrip().splitlines()[-TAIL:]))
    print(f"\n{len(results) - len(failed)} of {len(results)} scripts passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
