// Build valid OpenZL frames around fuzzed geozl headers and streams. This reaches
// decoder bindings more often than mutating complete frames.

#include "geozl/geozl.h"

#include "openzl/zl_common_types.h" // ZL_TernaryParam_disable
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_data.h"
#include "openzl/zl_decompress.h"
#include "openzl/zl_errors.h"
#include "openzl/zl_input.h"
#include "openzl/zl_output.h"
#include "openzl/zl_version.h" // ZL_MAX_FORMAT_VERSION
#include "openzl/codecs/zl_store.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Covers binoffset's largest header: 1 + 255 * (8 + 1).
#define MAX_HDR 4096
#define MAX_OUT 2
#define MAX_ELTS 8192
#define MAX_BYTES (MAX_ELTS * 8)
#define MAX_FRAME (1u << 20)

// Every geozl CTid and its input stream count.
typedef struct {
  uint32_t ctid;
  unsigned nbOut;
} codec_slot;

static const codec_slot kSlots[] = {
    {GEOZL_CTID_DELTA_W, 1},      {GEOZL_CTID_DELTA_N, 1},
    {GEOZL_CTID_PLANAR, 1},       {GEOZL_CTID_MED, 1},
    {GEOZL_CTID_AVERAGE, 1},      {GEOZL_CTID_WP_STATIC, 1},
    {GEOZL_CTID_QUANT_LINEAR, 1}, {GEOZL_CTID_QUANT_LOG, 1},
    {GEOZL_CTID_QUANT_SQRT, 1},   {GEOZL_CTID_DEINTERLEAVE, 2},
    {GEOZL_CTID_NODATA, 2},       {GEOZL_CTID_BINOFFSET, 2},
    {GEOZL_CTID_INTMULT, 2},      {GEOZL_CTID_FLOATQUANT, 2},
    {GEOZL_CTID_FLOATMULT, 2},
};

#define NB_SLOTS (sizeof(kSlots) / sizeof(kSlots[0]))

static struct {
  uint8_t hdr[MAX_HDR];
  size_t hdrSize;
  size_t nbOut;
  size_t eltWidth[MAX_OUT];
  size_t nbElts[MAX_OUT];
  const uint8_t *pay;
  size_t payLen;
  size_t payAt;
} g_plan;

typedef struct {
  const uint8_t *p;
  size_t n, i;
} bits;

static uint64_t take(bits *b, size_t k) {
  uint64_t v = 0;
  for (size_t j = 0; j < k; ++j)
    v |= (uint64_t)(b->i < b->n ? b->p[b->i++] : 0) << (8 * j);
  return v;
}

// Zero-pad short payloads.
static void fill(uint8_t *dst, size_t want) {
  const size_t have = g_plan.payAt < g_plan.payLen
                          ? g_plan.payLen - g_plan.payAt
                          : 0;
  const size_t n = have < want ? have : want;
  if (n)
    memcpy(dst, g_plan.pay + g_plan.payAt, n);
  if (n < want)
    memset(dst + n, 0, want - n);
  g_plan.payAt += n;
}

static ZL_Report stub_encode(ZL_Encoder *ectx, const ZL_Input *in) {
  (void)in;
  for (size_t o = 0; o < g_plan.nbOut; ++o) {
    ZL_Output *out = ZL_Encoder_createTypedStream(
        ectx, (int)o, g_plan.nbElts[o], g_plan.eltWidth[o]);
    if (out == NULL)
      return ZL_returnError(ZL_ErrorCode_allocation);
    fill((uint8_t *)ZL_Output_ptr(out), g_plan.nbElts[o] * g_plan.eltWidth[o]);
    ZL_Report r = ZL_Output_commit(out, g_plan.nbElts[o]);
    if (ZL_isError(r))
      return r;
  }
  ZL_Encoder_sendCodecHeader(ectx, g_plan.hdr, g_plan.hdrSize);
  return ZL_returnValue(g_plan.nbOut);
}

static size_t build_frame(const codec_slot *slot, uint8_t *dst, size_t cap) {
  static const ZL_Type kOut1[] = {ZL_Type_numeric};
  static const ZL_Type kOut2[] = {ZL_Type_numeric, ZL_Type_numeric};

  ZL_TypedEncoderDesc desc = {0};
  desc.gd.CTid = slot->ctid;
  desc.gd.inStreamType = ZL_Type_numeric;
  desc.gd.outStreamTypes = slot->nbOut == 1 ? kOut1 : kOut2;
  desc.gd.nbOutStreams = slot->nbOut;
  desc.transform_f = stub_encode;
  desc.name = "geozl_fuzz_stub";

  ZL_Compressor *c = ZL_Compressor_create();
  if (c == NULL)
    return 0;

  size_t written = 0;
  ZL_NodeID node = ZL_Compressor_registerTypedEncoder(c, &desc);
  if (!ZL_NodeID_isValid(node))
    goto done;

  ZL_GraphID g =
      slot->nbOut == 1
          ? ZL_Compressor_registerStaticGraph_fromNode(
                c, node, ZL_GRAPHLIST(ZL_GRAPH_STORE))
          : ZL_Compressor_registerStaticGraph_fromNode(
                c, node, ZL_GRAPHLIST(ZL_GRAPH_STORE, ZL_GRAPH_STORE));
  if (!ZL_GraphID_isValid(g))
    goto done;
  if (ZL_isError(ZL_Compressor_selectStartingGraphID(c, g)))
    goto done;

  ZL_CCtx *cctx = ZL_CCtx_create();
  if (cctx == NULL)
    goto done;
  if (ZL_isError(ZL_CCtx_refCompressor(cctx, c)) ||
      ZL_isError(ZL_CCtx_setParameter(cctx, ZL_CParam_formatVersion,
                                      ZL_MAX_FORMAT_VERSION))) {
    ZL_CCtx_free(cctx);
    goto done;
  }

  static const uint8_t seed[64] = {0};
  ZL_TypedRef *ref = ZL_TypedRef_createNumeric(seed, 1, sizeof(seed));
  if (ref != NULL) {
    ZL_Report r = ZL_CCtx_compressTypedRef(cctx, dst, cap, ref);
    ZL_TypedRef_free(ref);
    if (!ZL_isError(r))
      written = ZL_validResult(r);
  }
  ZL_CCtx_free(cctx);

done:
  ZL_Compressor_free(c);
  return written;
}

// -DGEOZL_FUZZ_VERBOSE prints decoder results for harness triage.
#ifdef GEOZL_FUZZ_VERBOSE
#include <stdio.h>
static void report(ZL_DCtx *dctx, ZL_Report r) {
  const char *why =
      ZL_isError(r) ? ZL_DCtx_getErrorContextString(dctx, r) : "(decoded)";
  char line[512];
  snprintf(line, sizeof(line), "%s", why != NULL ? why : "(null)");
  for (char *p = line; *p != '\0'; ++p)
    if (*p == '\n' || *p == '\t')
      *p = ' ';
  fprintf(stderr, "geozl-fuzz: %s\n", line);
}
#else
#define report(dctx, r) ((void)0)
#endif

static size_t decode(const uint8_t *frame, size_t size, uint8_t *dst,
                     size_t cap, int *ok) {
  *ok = 0;
  ZL_DCtx *dctx = ZL_DCtx_create();
  if (dctx == NULL)
    return 0;
  size_t out = 0;
  if (!ZL_isError(geozl_register_decoders(dctx)) &&
      !ZL_isError(ZL_DCtx_setParameter(dctx, ZL_DParam_checkContentChecksum,
                                       ZL_TernaryParam_disable))) {
    ZL_OutputInfo info;
    ZL_Report r = ZL_DCtx_decompressTyped(dctx, &info, dst, cap, frame, size);
    report(dctx, r);
    if (!ZL_isError(r)) {
      out = (size_t)info.decompressedByteSize;
      *ok = 1;
    }
  }
  ZL_DCtx_free(dctx);
  return out;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 8)
    return 0;
  bits b = {data, size, 0};

  const codec_slot *slot = &kSlots[take(&b, 1) % NB_SLOTS];

  memset(&g_plan, 0, sizeof(g_plan));
  g_plan.nbOut = slot->nbOut;

  g_plan.hdrSize = (size_t)take(&b, 2) % (MAX_HDR + 1);
  for (size_t i = 0; i < g_plan.hdrSize; ++i)
    g_plan.hdr[i] = (uint8_t)take(&b, 1);

  static const size_t kWidths[4] = {1, 2, 4, 8};
  size_t bytes = 0;
  for (size_t o = 0; o < g_plan.nbOut; ++o) {
    g_plan.eltWidth[o] = kWidths[take(&b, 1) & 3];
    g_plan.nbElts[o] = (size_t)take(&b, 2) % (MAX_ELTS + 1);
    bytes += g_plan.nbElts[o] * g_plan.eltWidth[o];
  }
  if (bytes > MAX_BYTES)
    return 0;

  g_plan.pay = data + b.i;
  g_plan.payLen = b.i < size ? size - b.i : 0;
  g_plan.payAt = 0;

  uint8_t *frame = malloc(MAX_FRAME);
  if (frame == NULL)
    return 0;
  const size_t frameSize = build_frame(slot, frame, MAX_FRAME);
  if (frameSize == 0) {
    free(frame);
    return 0;
  }

  const size_t cap = MAX_BYTES + 64;
  uint8_t *out = malloc(cap);
  if (out == NULL) {
    free(frame);
    return 0;
  }
  memset(out, 0x5A, cap);

  int ok = 0;
  const size_t n = decode(frame, frameSize, out, cap, &ok);

  // Successful decodes must not depend on the output buffer's initial bytes.
  if (ok) {
    uint8_t *again = malloc(cap);
    if (again != NULL) {
      memset(again, 0xA5, cap);
      int ok2 = 0;
      const size_t n2 = decode(frame, frameSize, again, cap, &ok2);
      if (!ok2 || n2 != n || (n && memcmp(again, out, n) != 0))
        abort();
      free(again);
    }
  }

  free(out);
  free(frame);
  return 0;
}
