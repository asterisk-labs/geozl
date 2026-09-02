#include "encode_blocked_transpose_zstd_binding.h"

#include "blocked_transpose_zstd.h"
#include "encode_blocked_transpose_zstd_kernel.h"

#include "common/endian.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_localParams.h"
#include "openzl/zl_output.h"

#ifndef ZSTD_STATIC_LINKING_ONLY
#define ZSTD_STATIC_LINKING_ONLY
#endif
#include <zstd.h>

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

enum { BTZ_OK = 0, BTZ_INVALID = 1, BTZ_ALLOC = 2, BTZ_ZSTD = 3 };

static int width_ok(size_t eltWidth) {
  return eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8;
}

static size_t normalized_block_size(size_t blockSize, size_t eltWidth) {
  if (!width_ok(eltWidth) || blockSize < eltWidth ||
      blockSize > BLOCKED_TRANSPOSE_ZSTD_MAX_BLOCK_SIZE)
    return 0;
  return blockSize - blockSize % eltWidth;
}

GEOZL_API size_t geozl_blocked_transpose_zstd_bound(size_t nbElts,
                                                    size_t eltWidth,
                                                    size_t blockSize) {
  blockSize = normalized_block_size(blockSize, eltWidth);
  if (blockSize == 0 || nbElts > SIZE_MAX / eltWidth)
    return 0;
  size_t left = nbElts * eltWidth;
  size_t bound = 0;
  while (left != 0) {
    const size_t rawSize = left < blockSize ? left : blockSize;
    const size_t frameBound = ZSTD_compressBound(rawSize);
    if (frameBound > UINT32_MAX || bound > SIZE_MAX - 4 - frameBound)
      return 0;
    bound += 4 + frameBound;
    left -= rawSize;
  }
  return bound;
}

static int encode_with_cctx(ZSTD_CCtx *cctx, void *scratch, void *dst0,
                            size_t dstCapacity, size_t *outSize,
                            const void *src0, size_t nbElts, size_t eltWidth,
                            size_t blockSize, int compressionLevel) {
  if (outSize == NULL || cctx == NULL)
    return BTZ_INVALID;
  *outSize = 0;
  blockSize = normalized_block_size(blockSize, eltWidth);
  const size_t bound = geozl_blocked_transpose_zstd_bound(
      nbElts, eltWidth, blockSize);
  if (blockSize == 0 || nbElts > SIZE_MAX / eltWidth ||
      (nbElts != 0 && (bound == 0 || dst0 == NULL || src0 == NULL ||
                       dstCapacity < bound)) ||
      (nbElts != 0 && eltWidth > 1 && scratch == NULL))
    return BTZ_INVALID;

  size_t zr = ZSTD_CCtx_reset(cctx, ZSTD_reset_session_and_parameters);
  if (ZSTD_isError(zr))
    return BTZ_ZSTD;
  zr = ZSTD_CCtx_setParameter(cctx, ZSTD_c_format, ZSTD_f_zstd1_magicless);
  if (ZSTD_isError(zr))
    return BTZ_ZSTD;
  zr = ZSTD_CCtx_setParameter(cctx, ZSTD_c_contentSizeFlag, 0);
  if (ZSTD_isError(zr))
    return BTZ_ZSTD;
  zr = ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 0);
  if (ZSTD_isError(zr))
    return BTZ_ZSTD;
  zr = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel,
                              compressionLevel);
  if (ZSTD_isError(zr))
    return BTZ_ZSTD;

  uint8_t *dst = (uint8_t *)dst0;
  const uint8_t *src = (const uint8_t *)src0;
  const size_t totalSize = nbElts * eltWidth;
  size_t at = 0;
  for (size_t rawAt = 0; rawAt < totalSize;) {
    const size_t left = totalSize - rawAt;
    const size_t rawSize = left < blockSize ? left : blockSize;
    const void *zsrc = src + rawAt;
    if (eltWidth > 1) {
      blocked_transpose_zstd_shuffle(scratch, zsrc, rawSize / eltWidth,
                                     eltWidth);
      zsrc = scratch;
    }
    const size_t frameBound = ZSTD_compressBound(rawSize);
    if (at > dstCapacity - 4 || frameBound > dstCapacity - at - 4)
      return BTZ_INVALID;
    const size_t cSize = ZSTD_compress2(cctx, dst + at + 4,
                                        dstCapacity - at - 4, zsrc, rawSize);
    if (ZSTD_isError(cSize) || cSize > UINT32_MAX)
      return BTZ_ZSTD;
    geozl_st_le32(dst + at, (uint32_t)cSize);
    at += 4 + cSize;
    rawAt += rawSize;
  }
  *outSize = at;
  return BTZ_OK;
}

GEOZL_API int geozl_blocked_transpose_zstd_encode(
    void *dst, size_t dstCapacity, size_t *outSize, const void *src,
    size_t nbElts, size_t eltWidth, size_t blockSize, int compressionLevel) {
  const size_t normalized = normalized_block_size(blockSize, eltWidth);
  if (normalized == 0 || nbElts > SIZE_MAX / eltWidth)
    return BTZ_INVALID;
  const size_t totalSize = nbElts * eltWidth;
  void *scratch = NULL;
  if (totalSize != 0 && eltWidth > 1) {
    const size_t scratchSize = totalSize < normalized ? totalSize : normalized;
    scratch = malloc(scratchSize);
    if (scratch == NULL)
      return BTZ_ALLOC;
  }
  ZSTD_CCtx *cctx = ZSTD_createCCtx();
  if (cctx == NULL) {
    free(scratch);
    return BTZ_ALLOC;
  }
  const int rc = encode_with_cctx(cctx, scratch, dst, dstCapacity, outSize, src,
                                  nbElts, eltWidth, normalized,
                                  compressionLevel);
  ZSTD_freeCCtx(cctx);
  free(scratch);
  return rc;
}

ZL_Report EI_geozl_blocked_transpose_zstd(ZL_Encoder *eictx,
                                           const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  const ZL_IntParam blockParam = ZL_Encoder_getLocalIntParam(
      eictx, BLOCKED_TRANSPOSE_ZSTD_PARAM_BLOCK_SIZE);
  const size_t blockSize = blockParam.paramValue > 0
                               ? (size_t)blockParam.paramValue
                               : 0;
  const size_t normalized = normalized_block_size(blockSize, eltWidth);
  const size_t bound = geozl_blocked_transpose_zstd_bound(
      nbElts, eltWidth, normalized);
  if (normalized == 0 || nbElts > SIZE_MAX / eltWidth ||
      (nbElts != 0 && bound == 0))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  uint8_t header[BLOCKED_TRANSPOSE_ZSTD_HEADER_SIZE] = {0};
  header[0] = BLOCKED_TRANSPOSE_ZSTD_VERSION;
  header[1] = (uint8_t)eltWidth;
  geozl_st_le32(header + 4, (uint32_t)normalized);
  geozl_st_le64(header + 8, (uint64_t)nbElts);
  ZL_Encoder_sendCodecHeader(eictx, header, sizeof(header));

  ZL_Output *out = ZL_Encoder_createTypedStream(eictx, 0, bound ? bound : 1, 1);
  ZL_ERR_IF_NULL(out, allocation);
  size_t written = 0;
  const int rc = geozl_blocked_transpose_zstd_encode(
      ZL_Output_ptr(out), bound, &written, ZL_Input_ptr(in), nbElts, eltWidth,
      normalized,
      ZL_Encoder_getCParam(eictx, ZL_CParam_compressionLevel));
  if (rc != BTZ_OK)
    return ZL_returnError(rc == BTZ_ALLOC ? ZL_ErrorCode_allocation
                                         : ZL_ErrorCode_transform_executionFailure);
  ZL_ERR_IF_ERR(ZL_Output_commit(out, written));
  return ZL_returnSuccess();
}
