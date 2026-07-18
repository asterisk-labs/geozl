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

int geozl_2d_compress_c(const char* method, uint32_t width, double max_error, int dtype, int format_version, const void* src, size_t num_elts, size_t elt_width, void* dst, size_t dst_capacity, size_t* out_size);
"""

ffi = FFI()
ffi.cdef(_CDEF)

_LIB_GLOBS = ("*.so", "*.so.*", "*.dylib", "*.dll")


def _bundled(match, reject=None):
    lib_dir = Path(__file__).parent / "_lib"
    for pattern in _LIB_GLOBS:
        for path in sorted(lib_dir.glob(pattern)):
            if match in path.name and (reject is None or reject not in path.name):
                return str(path)
    return None


def _bundled_lib():
    return _bundled("geozl_kernels")


def _load_lib():
    candidate = (os.environ.get("GEOZL_LIB") or _bundled_lib()
                 or ctypes.util.find_library("geozl_kernels"))
    if candidate is None:
        raise OSError("libgeozl_kernels not found, set GEOZL_LIB to its path")
    return ffi.dlopen(candidate)


lib = _load_lib()


# geozl_2d_compress_c is in libgeozl (links OpenZL), a different shared object
# from geozl_kernels. Loaded lazily so importing geozl needs only the kernels.
_lib_full = None


def _load_lib_full():
    global _lib_full
    if _lib_full is None:
        candidate = (os.environ.get("GEOZL_FULL_LIB")
                     or _bundled("geozl", reject="kernels")
                     or ctypes.util.find_library("geozl"))
        if candidate is None:
            raise OSError("libgeozl not found, set GEOZL_FULL_LIB to its path")
        _lib_full = ffi.dlopen(candidate)
    return _lib_full


def _ptr(buf):
    return ffi.from_buffer(buf)