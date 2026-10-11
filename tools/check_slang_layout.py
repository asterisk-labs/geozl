#!/usr/bin/env python3

import json
import sys

LAYOUTS = {
    "pfor": [
        "payload",
        "blocks",
        "planes",
        "output",
        "outputBytes",
        "rawValues",
        "rawCopies",
    ],
    "pivco": [
        "payload",
        "symbols",
        "trees",
        "nodes",
        "pivcoBlocks",
        "lanes",
        "laneBytes",
        "laneBlocks",
        "joins",
        "planes",
        "output",
        "outputBytes",
        "scratch",
        "rawValues",
        "rawCopies",
    ],
}


def binding_index(binding: dict[str, object]) -> int:
    index = binding.get("index")
    if isinstance(index, int):
        return index
    offset = binding.get("offset")
    if (
        binding.get("kind") == "uniform"
        and binding.get("size") == 16
        and isinstance(offset, int)
        and offset % 16 == 0
    ):
        return offset // 16
    raise ValueError(f"unsupported resource binding: {binding}")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] not in LAYOUTS:
        print("usage: check_slang_layout.py reflection.json pfor|pivco")
        return 2

    with open(sys.argv[1], encoding="utf-8") as file:
        reflection = json.load(file)
    actual = [
        (parameter["name"], binding_index(parameter["binding"]))
        for parameter in reflection["parameters"]
    ]
    expected = [
        (name, index) for index, name in enumerate(LAYOUTS[sys.argv[2]])
    ]
    if actual != expected:
        print(f"unexpected {sys.argv[2]} buffer layout: {actual}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
