from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_LIB_GLOBS = ("*.so", "*.so.*", "*.dylib", "*.dll")


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        lib_dir = Path(self.root) / "geozl" / "_lib"
        found = [p for g in _LIB_GLOBS for p in lib_dir.glob(g)]
        if not found:
            raise RuntimeError(
                f"no native library in {lib_dir}. Build it with `make lib` "
                "from a checkout. An sdist carries no C sources."
            )

        # Binary inside, so tag the wheel for this platform, not py3-none-any.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True