#!/usr/bin/env python3
"""Compile the Slang GPU programs into the modules core/src/gpu/modules embeds.

Metal source builds anywhere slangc runs. PTX needs slangc's NVRTC, and the
driver only loads PTX from an NVRTC no newer than itself, so build it with the
oldest CUDA the library should support.

Each build records the compiler inputs and every generated module in
OUTDIR/sources.sha256. --check fails when an input or output is stale.
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

SLANG_VERSION = "2026.19"
ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNELS = ROOT / "gpu" / "kernels"
PROGRAMS = KERNELS / "programs"
MODULE_DIR = ROOT / "core" / "src" / "gpu" / "modules"
MANIFEST = "sources.sha256"
TARGETS = ("metal", "ptx")
FIXED_FLAGS = ("-line-directive-mode", "none")
CHECK = ROOT / "tools" / "check_slang_layout.py"
SCAN = ["scan_planes", "scan_rows", "scan_columns"]
MODULES = {
    "pfor": (["unpack_blocks", *SCAN, "copy_raw"], [1, 2]),
    "pivco": (
        ["decode_symbols", "unpack_lane_blocks", "join_lanes", *SCAN,
         "copy_raw"],
        [1, 2],
    ),
    "place": (["place"], [1, 2, 4, 8, 16]),
}


def module_names(target: str) -> list[str]:
    return [f"{name}{width}.{target}"
            for name, (_, widths) in MODULES.items() for width in widths]


def file_digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources_digest(target: str) -> str:
    digest = hashlib.sha256(SLANG_VERSION.encode())
    digest.update(target.encode())
    digest.update(json.dumps(MODULES, sort_keys=True).encode())
    digest.update(json.dumps(FIXED_FLAGS).encode())
    for path in sorted(KERNELS.rglob("*.slang")):
        data = path.read_bytes()
        name = path.relative_to(ROOT).as_posix()
        digest.update(f"{name}\0{len(data)}\0".encode() + data)
    return digest.hexdigest()


def read_manifest(out: pathlib.Path) -> dict[str, str]:
    try:
        lines = (out / MANIFEST).read_text().splitlines()
    except FileNotFoundError:
        return {}
    return {target: digest for target, _, digest in
            (line.partition(" ") for line in lines)}


def check() -> int:
    recorded = read_manifest(MODULE_DIR)
    stale = []
    for target in TARGETS:
        if recorded.get(f"{target}:sources") != sources_digest(target):
            stale.append(target)
            continue
        for name in module_names(target):
            path = MODULE_DIR / name
            try:
                digest = file_digest(path)
            except FileNotFoundError:
                stale.append(target)
                break
            if recorded.get(f"{target}:{name}") != digest:
                stale.append(target)
                break
    if stale:
        print(f"stale {' and '.join(stale)} modules, rebuild them with "
              "tools/build_gpu_modules.py")
        return 1
    print("GPU modules are up to date")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--check"]:
        return check()
    if len(sys.argv) != 4 or sys.argv[2] not in TARGETS:
        print("usage: build_gpu_modules.py SLANGC metal|ptx OUTDIR | --check")
        return 2
    slangc, target, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    version = subprocess.run([slangc, "-version"], capture_output=True,
                             text=True, check=True)
    if (version.stdout + version.stderr).split()[0] != SLANG_VERSION:
        print(f"Slang {SLANG_VERSION} is required")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for name, (entries, widths) in MODULES.items():
            for width in widths:
                module = out / f"{name}{width}.{target}"
                reflection = pathlib.Path(scratch) / f"{module.name}.json"
                command = [slangc, str(PROGRAMS / f"{name}.slang")]
                for entry in entries:
                    command += ["-entry", entry, "-stage", "compute"]
                command += ["-target", target, "-D", f"ELT_BYTES={width}",
                            *FIXED_FLAGS,
                            "-reflection-json", str(reflection),
                            "-o", str(module)]
                subprocess.run(command, check=True)
                subprocess.run([sys.executable, str(CHECK), str(reflection),
                                name], check=True)
    recorded = read_manifest(out)
    for key in [key for key in recorded if key.startswith(f"{target}:")]:
        del recorded[key]
    recorded[f"{target}:sources"] = sources_digest(target)
    for name in module_names(target):
        recorded[f"{target}:{name}"] = file_digest(out / name)
    (out / MANIFEST).write_text(
        "".join(f"{key} {recorded[key]}\n" for key in sorted(recorded)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
