#include "decode_planar_zigzag_pivco_binding.h"
#include "decode_planar_zigzag_pivco_kernel.h"

#include "openzl/zl_data.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_errors_types.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"

// PivCo has no public kernel API; these come from the pinned OpenZL source.
#include "openzl/codecs/pivco_huffman/common_pivco_kernel.h"
#include "openzl/codecs/pivco_huffman/decode_pivco_kernel.h"

#include "common/endian.h"
#include "pfor/decode_pfor_kernel.h"
#include "pfor/pfor_check.h"

#include <assert.h>
#include <stdint.h>

typedef struct {
  uint8_t mode;
  size_t weightsSize;
  size_t bitsSize;
} LaneHeader;

typedef struct {
  uint64_t elementCount;
  size_t elementWidth;
  uint32_t rowWidth;
  uint32_t planes;
  uint32_t blockSize;
  LaneHeader lanes[8];
  size_t weightsSize;
  size_t bitsSize;
} CodecHeader;

static int lane_width_ok(size_t eltWidth) {
  return eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8;
}

static int read_header(CodecHeader *decoded, ZL_RBuffer encoded,
                       size_t weightsAvailable, size_t bitsAvailable) {
  if (encoded.size < PLANAR_ZIGZAG_PIVCO_HEADER_FIXED)
    return 1;

  const uint8_t *bytes = encoded.start;
  const size_t elementWidth = bytes[8];
  if (!lane_width_ok(elementWidth) ||
      encoded.size != PLANAR_ZIGZAG_PIVCO_HEADER_FIXED +
                          elementWidth * PLANAR_ZIGZAG_PIVCO_HEADER_LANE)
    return 1;

  *decoded = (CodecHeader){
      .elementCount = geozl_ld_le64(bytes),
      .elementWidth = elementWidth,
      .rowWidth = geozl_ld_le32(bytes + 9),
      .planes = geozl_ld_le32(bytes + 13),
      .blockSize = geozl_ld_le32(bytes + 17),
  };
  for (size_t laneIndex = 0; laneIndex < elementWidth; ++laneIndex) {
    const uint8_t *laneBytes = bytes + PLANAR_ZIGZAG_PIVCO_HEADER_FIXED +
                               laneIndex * PLANAR_ZIGZAG_PIVCO_HEADER_LANE;
    LaneHeader *lane = &decoded->lanes[laneIndex];
    lane->mode = laneBytes[0];
    lane->weightsSize = geozl_ld_le16(laneBytes + 1);
    lane->bitsSize = geozl_ld_le32(laneBytes + 3);
    if (lane->mode > PLANAR_ZIGZAG_PIVCO_LANE_PFOR ||
        lane->weightsSize > ZL_PIVCO_MAX_SYMBOLS ||
        (lane->mode == PLANAR_ZIGZAG_PIVCO_LANE_PFOR &&
         lane->weightsSize != 0) ||
        lane->weightsSize > weightsAvailable - decoded->weightsSize ||
        lane->bitsSize > bitsAvailable - decoded->bitsSize)
      return 1;
    decoded->weightsSize += lane->weightsSize;
    decoded->bitsSize += lane->bitsSize;
  }
  return decoded->weightsSize != weightsAvailable ||
         decoded->bitsSize != bitsAvailable;
}

static int element_count_is_bounded(const CodecHeader *header,
                                    const uint8_t *weights) {
  size_t weightsOffset = 0;
  for (size_t laneIndex = 0; laneIndex < header->elementWidth; ++laneIndex) {
    const LaneHeader *lane = &header->lanes[laneIndex];
    if (lane->mode == PLANAR_ZIGZAG_PIVCO_LANE_PFOR) {
      if (lane->bitsSize == 0 ||
          header->elementCount >
              (uint64_t)lane->bitsSize * GEOZL_PFOR_MAX_ELTS_PER_BYTE)
        return 0;
      continue;
    }

    size_t symbols = 0;
    for (size_t symbol = 0; symbol < lane->weightsSize; ++symbol)
      symbols += weights[weightsOffset + symbol] != 0;
    if (symbols == 0 ||
        (symbols > 1 && header->elementCount > (uint64_t)lane->bitsSize * 8))
      return 0;
    weightsOffset += lane->weightsSize;
  }
  return 1;
}

ZL_Report DI_geozl_planar_zigzag_pivco(ZL_Decoder *dictx,
                                       const ZL_Input *ins[]) {
  ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
  assert(ins != NULL);
  const ZL_Input *weightsIn = ins[0];
  const ZL_Input *bitsIn = ins[1];
  assert(weightsIn != NULL && bitsIn != NULL);
  assert(ZL_Input_type(weightsIn) == ZL_Type_numeric);
  assert(ZL_Input_type(bitsIn) == ZL_Type_serial);

  if (ZL_Input_eltWidth(weightsIn) != 1)
    return ZL_returnError(ZL_ErrorCode_corruption);

  const uint8_t *weights = ZL_Input_ptr(weightsIn);
  const uint8_t *bits = ZL_Input_ptr(bitsIn);
  const size_t weightsAvail = ZL_Input_numElts(weightsIn);
  const size_t bitsAvail = ZL_Input_numElts(bitsIn);
  CodecHeader header;
  if (read_header(&header, ZL_Decoder_getCodecHeader(dictx), weightsAvail,
                  bitsAvail) != 0)
    return ZL_returnError(ZL_ErrorCode_corruption);

  if (header.elementCount == 0) {
    if (header.weightsSize != 0 || header.bitsSize != 0)
      return ZL_returnError(ZL_ErrorCode_corruption);
    ZL_Output *empty =
        ZL_Decoder_create1OutStream(dictx, 0, header.elementWidth);
    ZL_ERR_IF_NULL(empty, allocation);
    ZL_ERR_IF_ERR(ZL_Output_commit(empty, 0));
    return ZL_returnSuccess();
  }

  if (header.planes == 0 || header.rowWidth == 0 ||
      header.elementCount % header.planes != 0)
    return ZL_returnError(ZL_ErrorCode_corruption);
  const uint64_t planeElts = header.elementCount / header.planes;
  if (header.rowWidth > planeElts || planeElts % header.rowWidth != 0)
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (header.blockSize == 0 || header.blockSize > ZL_PIVCO_MAX_BLOCK_SIZE)
    return ZL_returnError(ZL_ErrorCode_corruption);
  if (header.elementCount > (uint64_t)(SIZE_MAX / header.elementWidth))
    return ZL_returnError(ZL_ErrorCode_corruption);

  // A lane of two or more symbols spends at least a bit per value, in its root
  // bitmap or flat leaf. A lane of one symbol spends none, like OpenZL's own
  // constant codec, and is bounded by the caller's decompressed-size limit.
  // A PFOR lane spends at least two bytes per 256 values.
  if (!element_count_is_bounded(&header, weights))
    return ZL_returnError(ZL_ErrorCode_corruption);

  // The output first, so the decompressed-size limit applies before scratch.
  const size_t n = (size_t)header.elementCount;
  ZL_Output *out = ZL_Decoder_create1OutStream(dictx, n, header.elementWidth);
  ZL_ERR_IF_NULL(out, allocation);
  uint8_t *lanes = ZL_Decoder_getScratchSpace(dictx, n * header.elementWidth);
  ZL_ERR_IF_NULL(lanes, allocation);
  const size_t scratchSize =
      ZL_PivCoHuffmanDecode_scratchBytes(n, header.blockSize);
  uint8_t *scratch = ZL_Decoder_getScratchSpace(dictx, scratchSize);
  ZL_ERR_IF_NULL(scratch, allocation);

  size_t bitsOffset = 0;
  size_t weightsOffset = 0;
  for (size_t laneIndex = 0; laneIndex < header.elementWidth; ++laneIndex) {
    const LaneHeader *lane = &header.lanes[laneIndex];
    if (lane->mode == PLANAR_ZIGZAG_PIVCO_LANE_PFOR) {
      if (pfor_decode(lanes + laneIndex * n, n, 1, bits + bitsOffset,
                      lane->bitsSize) != 0)
        return ZL_returnError(ZL_ErrorCode_corruption);
    } else if (!ZL_PivCoHuffman_decode(
                   lanes + laneIndex * n, n, scratch, scratchSize,
                   weights + weightsOffset, lane->weightsSize,
                   bits + bitsOffset, lane->bitsSize, header.blockSize, NULL)) {
      return ZL_returnError(ZL_ErrorCode_corruption);
    }
    weightsOffset += lane->weightsSize;
    bitsOffset += lane->bitsSize;
  }

  if (planar_zigzag_pivco_unlanes(ZL_Output_ptr(out), lanes, header.rowWidth, n,
                                  header.elementWidth, header.planes) != 0)
    return ZL_returnError(ZL_ErrorCode_corruption);
  ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
  return ZL_returnSuccess();
}
