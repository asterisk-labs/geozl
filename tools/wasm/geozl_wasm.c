#include "geozl/geozl.h"

#include <emscripten/emscripten.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

EMSCRIPTEN_KEEPALIVE void *geozl_wasm_malloc(size_t size) {
  return malloc(size == 0 ? 1 : size);
}

EMSCRIPTEN_KEEPALIVE void geozl_wasm_free(void *ptr) { free(ptr); }

EMSCRIPTEN_KEEPALIVE int geozl_wasm_compress(
    const char *method, uint32_t width, uint32_t planes, const char *error,
    int dtype, int nodataMode, uint64_t nodataBits, const void *src,
    size_t numElts, size_t eltWidth, void *dst, size_t dstCapacity,
    size_t *outSize, char *errCtx, size_t errCtxSize) {
  return geozl_2d_compress_c(method, width, planes, error, dtype, nodataMode,
                             nodataBits, src, numElts, eltWidth, dst,
                             dstCapacity, outSize, errCtx, errCtxSize);
}

EMSCRIPTEN_KEEPALIVE size_t geozl_wasm_frame_dsize(const void *frame,
                                                   size_t frameSize) {
  return geozl_2d_frame_dsize_c(frame, frameSize);
}

EMSCRIPTEN_KEEPALIVE int geozl_wasm_decompress(
    const void *frame, size_t frameSize, void *dst, size_t dstCapacity,
    size_t *outSize, int verify, char *errCtx, size_t errCtxSize) {
  return geozl_2d_decompress_c(frame, frameSize, dst, dstCapacity, outSize,
                               verify, errCtx, errCtxSize);
}
