import ctypes.util
import os
from pathlib import Path

from cffi import FFI

_CDEF = """
size_t geozl_delta_w_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width,
                             size_t nb_elts, size_t elt_width);
void geozl_delta_w_decode_auto(void* dst, const void* src,
                           const uint8_t* header, size_t header_size,
                           size_t nb_elts, size_t elt_width);
size_t geozl_delta_n_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width,
                             size_t nb_elts, size_t elt_width);
void geozl_delta_n_decode_auto(void* dst, const void* src,
                           const uint8_t* header, size_t header_size,
                           size_t nb_elts, size_t elt_width);
size_t geozl_planar_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width,
                             size_t nb_elts, size_t elt_width);
void geozl_planar_decode_auto(void* dst, const void* src,
                           const uint8_t* header, size_t header_size,
                           size_t nb_elts, size_t elt_width);
size_t geozl_med_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width,
                             size_t nb_elts, size_t elt_width);
void geozl_med_decode_auto(void* dst, const void* src,
                           const uint8_t* header, size_t header_size,
                           size_t nb_elts, size_t elt_width);
size_t geozl_average_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width,
                             size_t nb_elts, size_t elt_width);
void geozl_average_decode_auto(void* dst, const void* src,
                           const uint8_t* header, size_t header_size,
                           size_t nb_elts, size_t elt_width);
size_t geozl_wp_static_encode_auto(void* dst, uint8_t* header, size_t header_cap,
                             const void* src, size_t width,
                             size_t nb_elts, size_t elt_width);
void geozl_wp_static_decode_auto(void* dst, const void* src,
                           const uint8_t* header, size_t header_size,
                           size_t nb_elts, size_t elt_width);
void geozl_quant_linear_encode(void* dst, const void* src,
                               double scale, int dtype, size_t nb_elts);
void geozl_quant_linear_decode(void* dst, const void* src,
                               double scale, int dtype, size_t nb_elts);
"""

ffi = FFI()
ffi.cdef(_CDEF)

_LIB_GLOBS = ("*.so", "*.so.*", "*.dylib", "*.dll")


def _bundled_lib():
    lib_dir = Path(__file__).parent / "_lib"
    for pattern in _LIB_GLOBS:
        for path in sorted(lib_dir.glob(pattern)):
            return str(path)
    return None


def _load_lib():
    # GEOZL_LIB beats everything :D, then the wheel copy, then the OS path.
    candidate = (os.environ.get("GEOZL_LIB") or _bundled_lib()
                 or ctypes.util.find_library("geozl_kernels"))
    if candidate is None:
        raise OSError("libgeozl_kernels not found, set GEOZL_LIB to its path")
    return ffi.dlopen(candidate)


lib = _load_lib()


def _ptr(arr):
    return ffi.cast("void *", arr.ctypes.data)
