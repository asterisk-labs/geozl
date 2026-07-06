import ctypes.util
import os
from pathlib import Path

from cffi import FFI

# The kernels are the transform, one implementation shared with the C reader.
# Python calls them here, packs the codec header itself, and drives the graph
# through openzl.ext.
_CDEF = """
void delta_w_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void delta_w_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void delta_n_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void delta_n_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void planar_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void planar_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void med_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void med_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void average_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void average_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);

void wp_static_train(int16_t coeffs[4], uint8_t* shift, const void* src, size_t width, size_t nb_elts, size_t elt_width);
void wp_static_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width, const int16_t coeffs[4], uint8_t shift);
void wp_static_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width, const int16_t coeffs[4], uint8_t shift);

void deinterleave_split(void* out0, void* out1, const void* src, size_t nb_elts, size_t elt_width);
void deinterleave_join(void* dst, const void* in0, const void* in1, size_t nb_elts, size_t elt_width);

void quant_linear_encode(void* dst, const void* src, double scale, int dtype, size_t nb_elts);
void quant_linear_decode(void* dst, const void* src, double scale, int dtype, size_t nb_elts);
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
    candidate = (os.environ.get("GEOZL_LIB") or _bundled_lib()
                 or ctypes.util.find_library("geozl_kernels"))
    if candidate is None:
        raise OSError("libgeozl_kernels not found, set GEOZL_LIB to its path")
    return ffi.dlopen(candidate)


lib = _load_lib()


def _ptr(arr):
    return ffi.cast("void *", arr.ctypes.data)
