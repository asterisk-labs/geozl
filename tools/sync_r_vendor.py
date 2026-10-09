"""Copy the native sources the R package compiles into bindings/r/src/ext.

R builds a package from its own directory, so the GeoZL core and the OpenZL
pieces its CMake reaches are carried inside it. The R build configures them
with the same CMake files as every other build.
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "bindings" / "r"
# Short, so OpenZL's deepest file stays within tar's portable 100 bytes.
VENDOR = PACKAGE / "src" / "ext"
OPENZL = ROOT / "extern" / "openzl"


def copy(source: Path, destination: Path, *skip: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        # Dotfiles such as .clang-format are not sources, and R flags them;
        # the generator scripts in Python are not needed to build.
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".*", "*.py", *skip))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    if not (OPENZL / "deps" / "zstd" / "lib" / "zstd.h").exists():
        raise SystemExit("extern/openzl and its deps are missing, run make submodules")
    shutil.rmtree(VENDOR, ignore_errors=True)

    for part in ("CMakeLists.txt", "include", "src"):
        copy(ROOT / "core" / part, VENDOR / "core" / part)

    for part in ("CMakeLists.txt", "LICENSE", "build-scripts", "include", "src"):
        copy(OPENZL / part, VENDOR / "openzl" / part)
    # tools/ is always added. Its subdirectories build nothing with the tools
    # off, except fileio, whose library is declared unconditionally.
    copy(OPENZL / "tools" / "CMakeLists.txt",
         VENDOR / "openzl" / "tools" / "CMakeLists.txt")
    for listfile in (OPENZL / "tools").glob("*/CMakeLists.txt"):
        copy(listfile, VENDOR / "openzl" / listfile.relative_to(OPENZL))
    copy(OPENZL / "tools" / "fileio", VENDOR / "openzl" / "tools" / "fileio")
    # The top level always declares the profile graph library too.
    for name in ("profile_graphs.cpp", "profile_graphs.h"):
        copy(OPENZL / "cli" / "utils" / name,
             VENDOR / "openzl" / "cli" / "utils" / name)
    for dep in ("zstd", "lz4"):
        # lz4's CMake reads its README for CPack.
        for part in ("lib", "build/cmake", "LICENSE", "COPYING", "README.md"):
            if (OPENZL / "deps" / dep / part).exists():
                # Their own GNU Makefiles go unused, and R flags them.
                copy(OPENZL / "deps" / dep / part,
                     VENDOR / "openzl" / "deps" / dep / part, "Makefile", "dll")

    # Binding package versions follow VERSION, like the wheel's.
    version = (ROOT / "VERSION").read_text().strip()
    versions = (
        (PACKAGE / "DESCRIPTION", r"(?m)^Version: .*$", f"Version: {version}"),
        (ROOT / "bindings" / "julia" / "Project.toml",
         r'(?m)^version = ".*"$', f'version = "{version}"'),
    )
    for path, pattern, replacement in versions:
        text = path.read_text()
        updated, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise SystemExit(f"cannot find the version in {path}")
        if updated != text:
            path.write_text(updated)

    licenses = PACKAGE / "inst" / "licenses"
    shutil.rmtree(licenses, ignore_errors=True)
    for name in ("LICENSE.OpenZL", "LICENSE.Zstandard", "LICENSE.LZ4"):
        copy(ROOT / "licenses" / name, licenses / name)


if __name__ == "__main__":
    main()
