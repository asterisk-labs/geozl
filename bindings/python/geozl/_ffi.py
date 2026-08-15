import ctypes.util
import os
from pathlib import Path

from cffi import FFI

# The kernels are the transform, one implementation shared with the C reader.
# Python calls them here, packs the codec header itself, and drives the graph
# through openzl.ext.
_CDEF = """
int delta_w_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int delta_w_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int delta_n_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int delta_n_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int planar_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int planar_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int med_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int med_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int average_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int average_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width);

int wp_static_train(int16_t coeffs[4], uint8_t* shift, const void* src, size_t width, size_t nb_elts, size_t elt_width);
int wp_static_encode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width, const int16_t coeffs[4], uint8_t shift);
int wp_static_decode(void* dst, const void* src, size_t width, size_t nb_elts, size_t elt_width, const int16_t coeffs[4], uint8_t shift);

void deinterleave_split(void* out0, void* out1, const void* src, size_t nb_elts, size_t elt_width);
void deinterleave_join(void* dst, const void* in0, const void* in1, size_t nb_elts, size_t elt_width);

int nodata_find_nan(uint64_t* pattern, const void* src, size_t nb_elts, size_t elt_width);
void nodata_mark_nan(uint8_t* mask, const void* src, size_t nb_elts, size_t elt_width);
void nodata_mark_value(uint8_t* mask, const void* src, size_t nb_elts, size_t elt_width, uint64_t pattern);
void nodata_fill(void* dst, const void* src, const uint8_t* mask, size_t width, size_t nb_elts, size_t elt_width);
void nodata_restore(void* dst, const void* values, const uint8_t* mask, size_t nb_elts, size_t elt_width, uint64_t pattern);

// The three quantizers. Scan and resolve differ per curve.
typedef struct { double max_error; unsigned char store; } quant_linear_spec;
typedef struct { double maxAbs; int anyNegative; } quant_linear_stats;
typedef struct { unsigned char flags; double step; } quant_linear_params;
int quant_linear_parse(const char* s, quant_linear_spec* out, char* err, size_t err_size);
int quant_linear_scan(const void* src, int dtype, size_t nb_elts, quant_linear_stats* out);
int quant_linear_resolve(const quant_linear_spec* sp, int dtype, const quant_linear_stats* sc, quant_linear_params* out, char* err, size_t err_size);
int quant_linear_encode(void* dst, const void* src, const quant_linear_params* p, int dtype, size_t nb_elts);
int quant_linear_decode(void* dst, const void* src, const quant_linear_params* p, int dtype, size_t nb_elts);

typedef struct { double rel_err; unsigned char store; } quant_log_spec;
typedef struct { unsigned char flags; double step; } quant_log_params;
typedef struct { double minAbs; double maxAbs; int anyNegative; int anySubnormal; } quant_log_stats;
int quant_log_parse(const char* s, quant_log_spec* out, char* err, size_t err_size);
int quant_log_scan(const void* src, int dtype, size_t nb_elts, quant_log_stats* out);
int quant_log_resolve(const quant_log_spec* sp, int dtype, const quant_log_stats* sc, quant_log_params* out, char* err, size_t err_size);
int quant_log_encode(void* dst, const void* src, const quant_log_params* p, int dtype, size_t nb_elts);
int quant_log_decode(void* dst, const void* src, const quant_log_params* p, int dtype, size_t nb_elts);

typedef struct { double k; double a; double b; unsigned char store; unsigned char have_ab; } quant_sqrt_spec;
typedef struct { unsigned char flags; double step; double offset; } quant_sqrt_params;
typedef struct { double lo; double hi; int anyNegative; int anyNonFinite; } quant_sqrt_stats;
typedef struct { double a; double b; int ok; int blocks; int bins; double range; double colin; double resid; } quant_sqrt_noise;
int quant_sqrt_parse(const char* s, quant_sqrt_spec* out, char* err, size_t err_size);
int quant_sqrt_scan(const void* src, int dtype, size_t nb_elts, quant_sqrt_stats* out);
int quant_sqrt_resolve(const quant_sqrt_spec* sp, int dtype, const quant_sqrt_stats* sc, const quant_sqrt_noise* ft, quant_sqrt_params* out, char* err, size_t err_size);
int quant_sqrt_encode(void* dst, const void* src, const quant_sqrt_params* p, int dtype, size_t nb_elts);
int quant_sqrt_decode(void* dst, const void* src, const quant_sqrt_params* p, int dtype, size_t nb_elts);

typedef struct { double* mu; double* s2; size_t n; size_t cap; size_t stride; size_t seen; int failed; } quant_sqrt_accum;
void quant_sqrt_accum_init(quant_sqrt_accum* acc);
void quant_sqrt_accum_free(quant_sqrt_accum* acc);
int quant_sqrt_accum_push(quant_sqrt_accum* acc, const void* src, int dtype, size_t width, size_t height);
int quant_sqrt_accum_solve(const quant_sqrt_accum* acc, quant_sqrt_noise* out, char* err, size_t err_size);

int geozl_2d_compress_c(const char* method, uint32_t width, uint32_t planes, const char* error, int dtype, int nodata_mode, uint64_t nodata_bits, const void* src, size_t num_elts, size_t elt_width, void* dst, size_t dst_capacity, size_t* out_size, char* err_ctx, size_t err_ctx_size);
size_t geozl_2d_frame_dsize_c(const void* frame, size_t frame_size);
int geozl_2d_decompress_c(const void* frame, size_t frame_size, void* dst, size_t dst_capacity, size_t* out_size, int verify, char* err_ctx, size_t err_ctx_size);
int geozl_2d_bench_c(const char* method, uint32_t width, uint32_t planes, const char* error, int dtype, int nodata_mode, uint64_t nodata_bits, const void* src, size_t num_elts, size_t elt_width, size_t reps, int checksum, int verify, size_t* comp_size, double* enc_sec, double* dec_sec, char* err_ctx, size_t err_ctx_size);
typedef struct geozl_2d_graph_s geozl_2d_graph;
int geozl_2d_graph_open_c(geozl_2d_graph** out, const char* method, uint32_t width, uint32_t planes, const char* error, int dtype, int nodata_mode, uint64_t nodata_bits, const void* src, size_t num_elts, size_t elt_width, char* err_ctx, size_t err_ctx_size);
int geozl_2d_compress_graph_c(geozl_2d_graph* g, const void* src, size_t num_elts, void* dst, size_t dst_capacity, size_t* out_size, char* err_ctx, size_t err_ctx_size);
void geozl_2d_graph_close_c(geozl_2d_graph* g);
int geozl_2d_grid_c(const char* method, size_t elt_width, char* names, size_t stride, size_t max_names, size_t* out_count);

unsigned geozl_simd_built(void);
unsigned geozl_simd_cpu(void);
int geozl_simd_active(void);
const char* geozl_simd_name(int path);
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


# The 2d entries live in libgeozl (links OpenZL), loaded lazily.
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
