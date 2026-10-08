#include "encode_planar_zigzag_pivco_binding.h"
#include "encode_planar_zigzag_pivco_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

// PivCo has no public kernel API; these come from the pinned OpenZL source.
#define HUF_STATIC_LINKING_ONLY
#include "openzl/codecs/pivco_huffman/common_pivco_kernel.h"
#include "openzl/codecs/pivco_huffman/encode_pivco_kernel.h"
#include "openzl/fse/huf.h"
#include "openzl/shared/histogram.h"

#include "common/endian.h"
#include "common/graph_num1to1.h"
#include "common/raster.h"
#include "pfor/encode_pfor_kernel.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

static int lane_width_ok(size_t eltWidth) {
  return eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8;
}

// Huffman weights for one lane, chosen as OpenZL's own PivCo encoder does.
// Returns the table log, or -1.
static int lane_weights(uint8_t weights[ZL_PIVCO_MAX_SYMBOLS],
                        size_t *weightsSize, const uint8_t *src, size_t n) {
  memset(weights, 0, ZL_PIVCO_MAX_SYMBOLS);
  ZL_Histogram8 hist;
  ZL_Histogram_init(&hist.base, 255);
  ZL_Histogram_build(&hist.base, src, n, 1);
  const unsigned maxSymbol = hist.base.maxSymbol;
  *weightsSize = (size_t)maxSymbol + 1;
  if (hist.base.cardinality == 1) {
    weights[maxSymbol] = 1;
    return 0;
  }
  HUF_CREATE_STATIC_CTABLE(ctable, HUF_SYMBOLVALUE_MAX);
  const unsigned log = HUF_optimalTableLog(ZL_PIVCO_MAX_TABLE_LOG, n, maxSymbol);
  const size_t built = HUF_buildCTable(ctable, hist.base.count, maxSymbol, log);
  if (HUF_isError(built) || built > ZL_PIVCO_MAX_TABLE_LOG)
    return -1;
  const unsigned tableLog = (unsigned)built;
  for (unsigned s = 0; s <= maxSymbol; ++s) {
    const unsigned bits = HUF_getNbBitsFromCTable(ctable, s);
    weights[s] = bits == 0 ? 0 : (uint8_t)(tableLog + 1 - bits);
  }
  return (int)tableLog;
}

ZL_Report EI_geozl_planar_zigzag_pivco(ZL_Encoder *eictx,
                                        const ZL_Input *in) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
  assert(in != NULL);
  assert(ZL_Input_type(in) == ZL_Type_numeric);

  const size_t eltWidth = ZL_Input_eltWidth(in);
  const size_t nbElts = ZL_Input_numElts(in);
  if (!lane_width_ok(eltWidth))
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  // The lane histogram counts in unsigned, as OpenZL's encoder does.
  if (nbElts > UINT_MAX)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  ZL_IntParam wp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_WIDTH);
  const uint32_t width = geozl_row_width_declared(
      (wp.paramId == GEOZL_PARAM_WIDTH) ? (uint32_t)wp.paramValue
                                        : (uint32_t)nbElts,
      nbElts);
  ZL_IntParam pp = ZL_Encoder_getLocalIntParam(eictx, GEOZL_PARAM_PLANES);
  const uint32_t planes = geozl_planes_declared(
      (pp.paramId == GEOZL_PARAM_PLANES) ? (uint32_t)pp.paramValue : 1u, width,
      nbElts);
  const size_t blockSize = ZL_PIVCO_DEFAULT_BLOCK_SIZE;

  uint8_t header[PLANAR_ZIGZAG_PIVCO_HEADER_MAX];
  const size_t headerSize =
      PLANAR_ZIGZAG_PIVCO_HEADER_FIXED + eltWidth * PLANAR_ZIGZAG_PIVCO_HEADER_LANE;
  memset(header, 0, sizeof(header));
  geozl_st_le64(header, (uint64_t)nbElts);
  header[8] = (uint8_t)eltWidth;
  geozl_st_le32(header + 9, width);
  geozl_st_le32(header + 13, planes);
  geozl_st_le32(header + 17, (uint32_t)blockSize);

  if (nbElts == 0) {
    ZL_Encoder_sendCodecHeader(eictx, header, headerSize);
    for (int i = 0; i < 2; ++i) {
      ZL_Output *empty = ZL_Encoder_createTypedStream(eictx, i, 0, 1);
      ZL_ERR_IF_NULL(empty, allocation);
      ZL_ERR_IF_ERR(ZL_Output_commit(empty, 0));
    }
    return ZL_returnSuccess();
  }

  if (nbElts > SIZE_MAX / eltWidth)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  uint8_t *lanes = ZL_Encoder_getScratchSpace(eictx, nbElts * eltWidth);
  ZL_ERR_IF_NULL(lanes, allocation);
  if (planar_zigzag_pivco_lanes(lanes, ZL_Input_ptr(in), width, nbElts,
                                eltWidth, planes) != 0)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);

  const size_t pivcoBound = ZL_PivCoHuffmanEncode_bound(nbElts, blockSize);
  const size_t pforBound = pfor_bound(nbElts, 1);
  if (pivcoBound == SIZE_MAX || pforBound == 0)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  const size_t laneBound = pivcoBound > pforBound ? pivcoBound : pforBound;
  if (laneBound > SIZE_MAX / eltWidth)
    return ZL_returnError(ZL_ErrorCode_node_invalid_input);
  uint8_t *pforTry = ZL_Encoder_getScratchSpace(eictx, pforBound);
  ZL_ERR_IF_NULL(pforTry, allocation);
  ZL_Output *weightsOut =
      ZL_Encoder_createTypedStream(eictx, 0, eltWidth * ZL_PIVCO_MAX_SYMBOLS, 1);
  ZL_ERR_IF_NULL(weightsOut, allocation);
  ZL_Output *bitsOut =
      ZL_Encoder_createTypedStream(eictx, 1, laneBound * eltWidth, 1);
  ZL_ERR_IF_NULL(bitsOut, allocation);
  const size_t scratchSize =
      ZL_PivCoHuffmanEncode_scratchElements(nbElts, blockSize);
  uint8_t *scratch = ZL_Encoder_getScratchSpace(eictx, scratchSize);
  ZL_ERR_IF_NULL(scratch, allocation);

  uint8_t *weightsDst = ZL_Output_ptr(weightsOut);
  uint8_t *bitsDst = ZL_Output_ptr(bitsOut);
  size_t weightsTotal = 0, bitsTotal = 0;
  for (size_t k = 0; k < eltWidth; ++k) {
    const uint8_t *lane = lanes + k * nbElts;
    uint8_t weights[ZL_PIVCO_MAX_SYMBOLS];
    size_t weightsSize = 0;
    const int tableLog = lane_weights(weights, &weightsSize, lane, nbElts);
    if (tableLog < 0)
      return ZL_returnError(ZL_ErrorCode_GENERIC);
    size_t used = ZL_PivCoHuffman_encode(
        bitsDst + bitsTotal, pivcoBound, scratch, scratchSize, weights,
        weightsSize, tableLog, lane, nbElts, blockSize, NULL);
    if (used == SIZE_MAX)
      return ZL_returnError(ZL_ErrorCode_GENERIC);
    uint8_t mode = PLANAR_ZIGZAG_PIVCO_LANE_PIVCO;
    size_t pforSize = 0;
    if (pfor_encode(pforTry, pforBound, &pforSize, lane, nbElts, 1) != 0)
      return ZL_returnError(ZL_ErrorCode_GENERIC);
    if (pforSize < used + weightsSize) {
      memcpy(bitsDst + bitsTotal, pforTry, pforSize);
      used = pforSize;
      weightsSize = 0;
      mode = PLANAR_ZIGZAG_PIVCO_LANE_PFOR;
    }
    if (used > UINT32_MAX)
      return ZL_returnError(ZL_ErrorCode_GENERIC);
    memcpy(weightsDst + weightsTotal, weights, weightsSize);
    uint8_t *h = header + PLANAR_ZIGZAG_PIVCO_HEADER_FIXED +
                 k * PLANAR_ZIGZAG_PIVCO_HEADER_LANE;
    h[0] = mode;
    geozl_st_le16(h + 1, (uint16_t)weightsSize);
    geozl_st_le32(h + 3, (uint32_t)used);
    weightsTotal += weightsSize;
    bitsTotal += used;
  }
  ZL_Encoder_sendCodecHeader(eictx, header, headerSize);
  ZL_ERR_IF_ERR(ZL_Output_commit(weightsOut, weightsTotal));
  ZL_ERR_IF_ERR(ZL_Output_commit(bitsOut, bitsTotal));
  return ZL_returnSuccess();
}
