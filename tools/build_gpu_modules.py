#!/usr/bin/env python3
"""Compile the Slang GPU programs into the modules core/src/gpu/modules embeds.

Metal source builds anywhere slangc runs. PTX needs slangc's NVRTC, and the
driver only loads PTX from an NVRTC no newer than itself, so build it with the
oldest CUDA the library should support.
"""

import pathlib
import subprocess
import sys
import tempfile

SLANG_VERSION = "2026.19"
ROOT = pathlib.Path(__file__).resolve().parent.parent
PROGRAMS = ROOT / "gpu" / "kernels" / "programs"
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


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[2] not in ("metal", "ptx"):
        print("usage: build_gpu_modules.py SLANGC metal|ptx OUTDIR")
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
                            "-line-directive-mode", "none",
                            "-reflection-json", str(reflection),
                            "-o", str(module)]
                subprocess.run(command, check=True)
                subprocess.run([sys.executable, str(CHECK), str(reflection),
                                name], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
