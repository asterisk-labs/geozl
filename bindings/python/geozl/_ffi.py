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

void binoffset_split(uint8_t* bins, void* offsets, const void* src, size_t nb_elts, size_t elt_width);
int  binoffset_join(void* dst, const uint8_t* bins, const void* offsets, size_t nb_elts, size_t elt_width);
void binoffset_split_table(uint8_t* bins, void* offsets, const void* src, size_t nb_elts, size_t elt_width, const uint64_t lowers[256], const uint8_t offset_bits[256], unsigned nb_bins);
int  binoffset_join_table(void* dst, const uint8_t* bins, const void* offsets, size_t nb_elts, size_t elt_width, const uint64_t lowers[256], const uint8_t offset_bits[256], unsigned nb_bins);
size_t binoffset_histogram(const uint64_t* sorted, size_t n, unsigned n_bins_log, uint64_t* out_lowers, uint64_t* out_uppers, uint32_t* out_counts);
unsigned binoffset_optimize_table(const uint64_t* fine_lowers, const uint64_t* fine_uppers, const uint32_t* fine_counts, size_t n_fine, double meta_bits, uint64_t* out_lowers, uint8_t* out_offset_bits, unsigned max_bins);
size_t binoffset_split_pack(uint8_t* bins, uint8_t* packed, const void* src, size_t nb_elts, size_t elt_width, const uint64_t lowers[256], const uint8_t offset_bits[256], unsigned nb_bins);
int  binoffset_unpack_join(void* dst, const uint8_t* bins, const uint8_t* packed, size_t packed_len, size_t nb_elts, size_t elt_width, const uint64_t lowers[256], const uint8_t offset_bits[256], unsigned nb_bins);

void intmult_split(void* mults, void* adjs, const void* src, size_t nb_elts, size_t elt_width, uint64_t base);
int  intmult_join(void* dst, const void* mults, const void* adjs, size_t nb_elts, size_t elt_width, uint64_t base);

void floatquant_split(void* primary, void* secondary, const void* src, size_t nb_elts, size_t elt_width, unsigned k);
int  floatquant_join(void* dst, const void* primary, const void* secondary, size_t nb_elts, size_t elt_width, unsigned k);

void floatmult_split(void* primary, void* secondary, const void* src, size_t nb_elts, size_t elt_width, double base, double inv_base);
int  floatmult_join(void* dst, const void* primary, const void* secondary, size_t nb_elts, size_t elt_width, double base);

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


def _ptr(arr):
    return ffi.cast("void *", arr.ctypes.data)