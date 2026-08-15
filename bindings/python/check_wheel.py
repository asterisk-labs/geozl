"""Check licence notices and native libraries in a wheel.

    python bindings/python/check_wheel.py dist/*.whl
"""

import sys
import zipfile
from pathlib import Path

NOTICES = ("LICENSE", "NOTICE", "LICENSE.OpenZL", "LICENSE.Zstandard",
           "LICENSE.LZ4")

LIB_SUFFIXES = (".so", ".dylib", ".dll")


def check(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
    bad = []

    dist_info = {n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA")}
    if len(dist_info) != 1:
        return [f"expected one dist-info, found {sorted(dist_info)}"]
    info = dist_info.pop()

    for notice in NOTICES:
        want = f"{info}/licenses/{notice}"
        if want not in names:
            bad.append(f"missing {want}")

    libs = [n for n in names
            if n.startswith("geozl/_lib/") and n.endswith(LIB_SUFFIXES)]
    if not libs:
        bad.append("no native library under geozl/_lib/")

    return bad


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    failed = False
    for arg in argv[1:]:
        path = Path(arg)
        bad = check(path)
        for line in bad:
            print(f"{path.name}: {line}", file=sys.stderr)
        if bad:
            failed = True
        else:
            print(f"{path.name}: notices and native library present")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
