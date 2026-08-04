from typing import TypedDict

from ._ffi import ffi, lib

_PATHS = (0, 1, 2, 3)  # GEOZL_SIMD_ in core/src/common/simd.h


class SimdInfo(TypedDict):
    built: list[str]
    cpu: list[str]
    active: str


def _name(path: int) -> str:
    return ffi.string(lib.geozl_simd_name(path)).decode("ascii")


def _names(mask: int) -> list[str]:
    return [_name(p) for p in _PATHS if mask & (1 << p)]


def simd_info() -> SimdInfo:
    """Vector paths this build carries, this machine can run, and is running.

    A path present in cpu and missing from built is a build that lost its fast
    path, which nothing else reveals since the frames are identical either way.
    Set GEOZL_SIMD before importing to cap active at scalar, sse2, avx2 or neon.
    """
    return {
        "built": _names(lib.geozl_simd_built()),
        "cpu": _names(lib.geozl_simd_cpu()),
        "active": _name(lib.geozl_simd_active()),
    }
