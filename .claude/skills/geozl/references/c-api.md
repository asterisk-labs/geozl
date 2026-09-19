# C API

Public headers live in `core/include/geozl/`; `geozl.h` includes the rest plus the
OpenZL headers it needs. Both examples below compile with
`-std=c11 -Wall -Wextra` and ran against a 0.17.0 build.

## Contents

1. Stability
2. What `geozl.h` declares
3. Error conventions
4. Example: high-level `geozl_2d_*` API
5. Example: geozl nodes inside your own OpenZL graph
6. Building and linking
7. The kernels library

## 1. Stability

- Since 0.13.0 the declarations in `geozl/geozl.h` are a **stable source API**:
  functions and parameter lists stay; new entry points may appear; breaking changes
  need a major release.
- The **ABI is not stable before 1.0**: recompile C applications after updating GeoZL or
  OpenZL.
- History of source breaks before 0.13: `planes` arguments added to five node builders
  and to `geozl_2d_compress_c`, `geozl_2d_graph_open_c`, `geozl_2d_bench_c` (0.12.0);
  `nodataBits` as `uint64_t` replacing a `double` (0.8.1); `verify` and `checksum`
  arguments (0.9.0).

## 2. What `geozl.h` declares

Decoders and CTids:

```c
ZL_Report geozl_register_decoders(ZL_DCtx *dctx);   // every geozl codec, legacy ones included
int geozl_owns_ctid(uint32_t ctid);
int geozl_ctid_is_lossy(uint32_t ctid);
```

Node builders (return `ZL_NODE_ILLEGAL` on failure; `width` in samples; invalid plane
layouts are treated as one plane):

```c
ZL_NodeID geozl_node_delta_w(ZL_Compressor *c, uint32_t width);
ZL_NodeID geozl_node_delta_n(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_planar(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_planar_zigzag(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_planar_zigzag_pfor(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_med(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_average(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_wp_static(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_deinterleave(ZL_Compressor *c);
ZL_NodeID geozl_node_pfor(ZL_Compressor *c);
ZL_NodeID geozl_node_blocked_transpose_zstd(ZL_Compressor *c, uint32_t blockSize); // WIP
ZL_NodeID geozl_node_binoffset(ZL_Compressor *c);                                  // legacy
ZL_NodeID geozl_node_intmult(ZL_Compressor *c, uint64_t base);                     // legacy
ZL_NodeID geozl_node_floatquant(ZL_Compressor *c, unsigned k);                     // legacy
ZL_NodeID geozl_node_floatmult(ZL_Compressor *c, double base);                     // legacy

// params must come from the spec parsers and resolvers (quant_*_parse, _scan, _resolve)
ZL_NodeID geozl_node_quant_linear(ZL_Compressor *c, const quant_linear_params *params, int dtype);
ZL_NodeID geozl_node_quant_log(ZL_Compressor *c, const quant_log_params *params, int dtype);
ZL_NodeID geozl_node_quant_sqrt(ZL_Compressor *c, const quant_sqrt_params *params, int dtype);

typedef enum { GEOZL_NODATA_NONE = 0, GEOZL_NODATA_NAN = 1, GEOZL_NODATA_VALUE = 2 } geozl_nodata_mode;
// VALUE mode needs the sample dtype and the maximum error of following stages.
// Use radius 0 for lossless and INFINITY when that error is unknown.
ZL_NodeID geozl_node_nodata(ZL_Compressor *c, uint32_t width, geozl_nodata_mode mode, uint64_t valueBits, int dtype, double radius);
```

High-level raster API (the same engine the Python `profile`, `graph`, `compress`,
`decompress` and `coeffs` call):

| Function | Purpose |
| --- | --- |
| `geozl_2d_compress_c(method, width, planes, error, dtype, nodataMode, nodataBits, src, numElts, eltWidth, dst, dstCapacity, &outSize, errCtx, errCtxSize)` | one-shot compress |
| `geozl_2d_graph_open_c(&g, method, width, planes, error, dtype, nodataMode, nodataBits, src, numElts, eltWidth, errCtx, errCtxSize)` | build a reusable graph; `src` fixes the lossy domain |
| `geozl_2d_compress_graph_c(g, src, numElts, dst, dstCapacity, &outSize, errCtx, errCtxSize)` | compress one tile |
| `geozl_2d_compress_coeffs_c(g, src, numElts, coeffs, coeffsSize, dst, ...)` | same, attaching a `geozl_coeffs_pack` blob |
| `geozl_2d_graph_close_c(g)` | free; accepts `NULL` |
| `geozl_2d_frame_dsize_c(frame, frameSize)` | decompressed byte size, 0 when unreadable |
| `geozl_2d_decompress_c(frame, frameSize, dst, dstCapacity, &outSize, verify, errCtx, errCtxSize)` | decode; `dst` must be 8-byte aligned |
| `geozl_2d_frame_coeffs_c(frame, frameSize, dst, dstCapacity, &outSize)` | copy the coefficient blob; `dst == NULL` queries the size; `-1` when absent |
| `geozl_2d_bench_c(method, width, planes, error, dtype, nodataMode, nodataBits, src, numElts, eltWidth, reps, checksum, verify, &compSize, &encSec, &decSec, errCtx, errCtxSize)` | time one recipe |
| `geozl_2d_grid_c(prior, eltWidth, names, stride, maxNames, &count)` | list recipe names for a prior (`""` for all, `"none"`) |

`error` is `NULL` or `""` for lossless, or a full recipe (`"LINEAR:MAX_ERROR=V"`,
`"LOG:MAX_ERROR=P%"`, `"SQRT:MAX_ERROR=KN[,A=..,B=..]"`). The C API has no numeric
shorthand. `dtype` is a `geozl_dtype` code (`dtype.h`); `nodataBits` holds the
sentinel's low `eltWidth` bytes and is read only for `GEOZL_NODATA_VALUE`.

Coefficient blobs (`coeffs.h`, no OpenZL dependency): `geozl_coeffs_size`,
`geozl_coeffs_pack`, `geozl_coeffs_parse` (returns 0 success, 1 absent or not geozl,
-1 invalid, -2 buffers too small with counts filled in).

## 3. Error conventions

- `geozl_2d_*` return `0` on success or a positive `ZL_ErrorCode`, with a message in
  `errCtx`. Observed codes with OpenZL 0.2: `21` invalid parameter (unknown method, bad
  recipe, lossy domain refusal), `30` graph invalid (recipe does not apply to the
  element width), `55` node precondition (plain `entropy` on 4 or 8 bytes, tile size not
  divisible by the row width). Match on the message, not the number.
- Node builders return `ZL_NODE_ILLEGAL`; check with `ZL_NodeID_isValid`.
- OpenZL calls return `ZL_Report`; check `ZL_isError`, read details with
  `ZL_CCtx_getErrorContextString` or `ZL_DCtx_getErrorContextString`.

## 4. Example: high-level `geozl_2d_*` API

```c
#include "geozl/geozl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  enum { H = 256, W = 256 };
  static uint16_t tile[H * W];
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      tile[y * W + x] = (uint16_t)(2000 + 8 * y + 5 * x);

  char err[256];
  geozl_2d_graph *g = NULL;
  int rc = geozl_2d_graph_open_c(&g, "planar>zigzag>pfor", W, 1,
                                 "LINEAR:MAX_ERROR=2", GEOZL_DT_U16,
                                 GEOZL_NODATA_NONE, 0, tile, H * W,
                                 sizeof(uint16_t), err, sizeof(err));
  if (rc != 0) { fprintf(stderr, "open: %s (code %d)\n", err, rc); return 1; }

  size_t cap = 1024 + sizeof(tile) + sizeof(tile) / 2;
  void *frame = malloc(cap);
  size_t fsize = 0;
  rc = geozl_2d_compress_graph_c(g, tile, H * W, frame, cap, &fsize, err, sizeof(err));
  if (rc != 0) { fprintf(stderr, "compress: %s (code %d)\n", err, rc); return 2; }

  size_t dsize = geozl_2d_frame_dsize_c(frame, fsize);
  uint16_t *back = malloc(dsize ? dsize : 1);   // malloc memory is suitably aligned
  size_t outSize = 0;
  rc = geozl_2d_decompress_c(frame, fsize, back, dsize, &outSize, 1, err, sizeof(err));
  if (rc != 0) { fprintf(stderr, "decompress: %s (code %d)\n", err, rc); return 3; }

  printf("frame %zu bytes, decoded %zu bytes\n", fsize, outSize);
  free(back);
  free(frame);
  geozl_2d_graph_close_c(g);
  return 0;
}
```

This links against `libgeozl` alone; the compiler still needs the OpenZL include
directory because `geozl.h` includes OpenZL headers.

## 5. Example: geozl nodes inside your own OpenZL graph

```c
#include "geozl/geozl.h"

#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_input.h"
#include "openzl/zl_version.h"
#include "openzl/codecs/zl_entropy.h"
#include "openzl/codecs/zl_zigzag.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// planar -> zigzag -> entropy on a uint16 raster; returns 0 when the round trip is exact
int roundtrip_u16(const uint16_t *tile, uint32_t width, size_t count) {
  ZL_Compressor *c = ZL_Compressor_create();
  ZL_NodeID planar = geozl_node_planar(c, width, 1);
  if (!ZL_NodeID_isValid(planar)) return 1;
  const ZL_NodeID nodes[2] = {planar, ZL_NODE_ZIGZAG};
  ZL_GraphID g = ZL_Compressor_registerStaticGraph_fromPipelineNodes1o(c, nodes, 2, ZL_GRAPH_ENTROPY);
  if (ZL_isError(ZL_Compressor_selectStartingGraphID(c, g))) return 2;

  ZL_CCtx *cctx = ZL_CCtx_create();
  if (ZL_isError(ZL_CCtx_refCompressor(cctx, c))) return 3;   // before setting parameters
  if (ZL_isError(ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION))) return 4;
  // A lossy node also needs ZL_CParam_contentChecksum set to ZL_TernaryParam_disable.

  size_t cap = 1024 + 2 * count * sizeof(uint16_t);
  void *frame = malloc(cap);
  ZL_TypedRef *in = ZL_TypedRef_createNumeric(tile, sizeof(uint16_t), count);
  ZL_Report r = ZL_CCtx_compressTypedRef(cctx, frame, cap, in);
  ZL_TypedRef_free(in);
  if (ZL_isError(r)) return 5;
  size_t fsize = ZL_validResult(r);

  ZL_DCtx *dctx = ZL_DCtx_create();
  if (ZL_isError(geozl_register_decoders(dctx))) return 6;
  ZL_Report ds = ZL_getDecompressedSize(frame, fsize);
  if (ZL_isError(ds)) return 7;
  uint16_t *back = malloc(ZL_validResult(ds));
  ZL_OutputInfo info;
  // geozl frames hold one numeric output, so use the typed call; dst must be 8-byte aligned
  r = ZL_DCtx_decompressTyped(dctx, &info, back, ZL_validResult(ds), frame, fsize);
  int rc = ZL_isError(r) || memcmp(back, tile, count * sizeof(uint16_t)) != 0;

  free(back);
  free(frame);
  ZL_DCtx_free(dctx);
  ZL_CCtx_free(cctx);
  ZL_Compressor_free(c);
  return rc;
}
```

This example calls OpenZL directly, so it must link OpenZL too: a shared `libgeozl`
hides its vendored copy. Through CMake, `geozl::geozl` brings OpenZL along. By hand from
a build tree it linked with `libgeozl` plus `core/build/openzl/libopenzl.a`,
`zstd_build/lib/libzstd.a`, `lz4_build/liblz4.a` and the C++ runtime (`-lc++` on macOS).

Useful OpenZL glue from `core/src/2d/2d.c`: `ZL_NODE_DELTA_INT`,
`ZL_NODE_CONVERT_NUM_TO_SERIAL` (before `ZL_GRAPH_ZSTD`),
`ZL_NODE_CONVERT_NUM_TO_STRUCT_LE` with `ZL_Compressor_registerTransposeSplitGraph(c, backend)`,
`ZL_GRAPH_FIELD_LZ`, `ZL_GRAPH_STORE` (after serial outputs such as `pfor`),
`ZL_Compressor_registerStaticGraph_fromNode(c, node, ZL_GRAPHLIST(a, b))` for two-output
nodes such as `nodata`.

## 6. Building and linking

CMake project `core/` (C11, CMake 3.21+). Options:

| Option | Default | Effect |
| --- | --- | --- |
| `GEOZL_BUILD_FULL` | `ON` | build `geozl` (links OpenZL); `OFF` builds only the kernels |
| `GEOZL_BUILD_KERNELS_SHARED` | `ON` | build `geozl_kernels` shared library (what Python loads) |
| `GEOZL_USE_SYSTEM_OPENZL` | `OFF` | `find_package(openzl)` instead of the `extern/openzl` submodule |
| `GEOZL_SANITIZE` | `OFF` | ASan and UBSan |
| `GEOZL_WERROR` | `OFF` | warnings as errors on geozl targets |
| `GEOZL_BUILD_FUZZERS` | `OFF` | libFuzzer harnesses (clang) |

- Standalone builds produce a shared `geozl`; a parent project that sets
  `BUILD_SHARED_LIBS=OFF` before `add_subdirectory` gets a static one. Link the
  `geozl::geozl` target; OpenZL is a public dependency.
- Only `GEOZL_API` symbols are exported; the vendored OpenZL is hidden inside a shared
  `libgeozl`.
- `make install PREFIX=/opt/geozl` installs the library and `include/geozl`.
- Keep `-ffp-contract=off` on any code that compiles geozl kernels: quantizer
  reconstructions must be bit identical on arm64 and x86-64, and FMA contraction breaks that.

## 7. The kernels library

`libgeozl_kernels` holds the pure C transforms (`planar_encode`, `pfor_encode`,
`quant_linear_parse`, `quant_sqrt_accum_push`, `geozl_coeffs_pack`, SIMD reporting, ...)
without OpenZL. The Python package calls it through cffi in ABI mode, which is why its
signatures are mirrored by hand in `bindings/python/geozl/_ffi.py` and checked by
`test_cdef.py`. Kernels return a nonzero `int` for geometry they refuse and validate
everything they compute on, because Python can call them without any frame or binding
in between.
