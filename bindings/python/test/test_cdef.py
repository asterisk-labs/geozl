"""The cdef in _ffi.py is a hand written copy of the C declarations, loaded in
ABI mode, so cffi verifies nothing and a drifted signature corrupts memory
instead of raising. This diffs the two. Needs the C sources, so it skips on an
installed package.
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_HEADERS = [_ROOT / "core" / "include" / "geozl" / "geozl.h"]
_HEADERS += sorted(p for p in (_ROOT / "core" / "include" / "geozl").glob("*.h")
                   if p.name != "geozl.h")
_HEADERS += sorted((_ROOT / "core" / "src").rglob("*.h"))

if not _HEADERS[0].exists():
    pytest.skip("no C sources next to the package", allow_module_level=True)

# A declaration, not a definition. The return type may end in stars, as in
# `const char *geozl_simd_name(int)`.
_DECL = re.compile(
    r"([A-Za-z_][\w \t]*?)\s*((?:\*\s*)*)(\w+)\s*\(([^;{)]*)\)\s*;", re.M)

# None of this reaches the ABI.
_DROP = re.compile(r"\b(GEOZL_API|restrict|__restrict|const|static|inline)\b")

_NOT_A_TYPE = {"return", "if", "while", "for", "switch", "sizeof", "typedef"}

# Public functions present when the C API becomes source-stable in 0.13. New
# functions may be added, but these names and signatures must remain available.
_PUBLIC_API_0_13 = r"""
ZL_Report geozl_register_decoders(ZL_DCtx *dctx);
int geozl_owns_ctid(uint32_t ctid);
int geozl_ctid_is_lossy(uint32_t ctid);
ZL_NodeID geozl_node_delta_w(ZL_Compressor *c, uint32_t width);
ZL_NodeID geozl_node_delta_n(ZL_Compressor *c, uint32_t width,
                             uint32_t planes);
ZL_NodeID geozl_node_planar(ZL_Compressor *c, uint32_t width,
                            uint32_t planes);
ZL_NodeID geozl_node_med(ZL_Compressor *c, uint32_t width, uint32_t planes);
ZL_NodeID geozl_node_average(ZL_Compressor *c, uint32_t width,
                             uint32_t planes);
ZL_NodeID geozl_node_wp_static(ZL_Compressor *c, uint32_t width,
                               uint32_t planes);
ZL_NodeID geozl_node_deinterleave(ZL_Compressor *c);
ZL_NodeID geozl_node_binoffset(ZL_Compressor *c);
ZL_NodeID geozl_node_intmult(ZL_Compressor *c, uint64_t base);
ZL_NodeID geozl_node_floatquant(ZL_Compressor *c, unsigned k);
ZL_NodeID geozl_node_floatmult(ZL_Compressor *c, double base);
ZL_NodeID geozl_node_quant_linear(ZL_Compressor *c,
                                  const quant_linear_params *params, int dtype);
ZL_NodeID geozl_node_quant_log(ZL_Compressor *c,
                               const quant_log_params *params, int dtype);
ZL_NodeID geozl_node_quant_sqrt(ZL_Compressor *c,
                                const quant_sqrt_params *params, int dtype);
ZL_NodeID geozl_node_nodata(ZL_Compressor *c, uint32_t width,
                            geozl_nodata_mode mode, uint64_t valueBits);
int geozl_2d_compress_c(const char *method, uint32_t width, uint32_t planes,
                        const char *error, int dtype, int nodataMode,
                        uint64_t nodataBits, const void *src, size_t numElts,
                        size_t eltWidth, void *dst, size_t dstCapacity,
                        size_t *outSize, char *errCtx, size_t errCtxSize);
int geozl_2d_graph_open_c(geozl_2d_graph **out, const char *method,
                          uint32_t width, uint32_t planes, const char *error,
                          int dtype, int nodataMode, uint64_t nodataBits,
                          const void *src, size_t numElts, size_t eltWidth,
                          char *errCtx, size_t errCtxSize);
int geozl_2d_compress_graph_c(geozl_2d_graph *g, const void *src,
                              size_t numElts, void *dst, size_t dstCapacity,
                              size_t *outSize, char *errCtx, size_t errCtxSize);
void geozl_2d_graph_close_c(geozl_2d_graph *g);
size_t geozl_2d_frame_dsize_c(const void *frame, size_t frameSize);
int geozl_2d_decompress_c(const void *frame, size_t frameSize, void *dst,
                          size_t dstCapacity, size_t *outSize, int verify,
                          char *errCtx, size_t errCtxSize);
int geozl_2d_bench_c(const char *method, uint32_t width, uint32_t planes,
                     const char *error, int dtype, int nodataMode,
                     uint64_t nodataBits, const void *src, size_t numElts,
                     size_t eltWidth, size_t reps, int checksum, int verify,
                     size_t *compSize, double *encSec, double *decSec,
                     char *errCtx, size_t errCtxSize);
int geozl_2d_grid_c(const char *method, size_t eltWidth, char *names,
                    size_t stride, size_t maxNames, size_t *outCount);
"""


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _tokens(text):
    """Stars as separate tokens, so `void *x` and `void* x` agree."""
    return re.sub(r"\*", " * ", _DROP.sub(" ", text)).split()


def _params(text):
    text = re.sub(r"\b(\w+)\s*\[[^\]]*\]", r"* \1", text)  # arr[4] -> * arr
    out = []
    for part in text.split(","):
        toks = _tokens(part)
        if not toks or toks == ["void"]:
            continue
        if len(toks) > 1 and re.fullmatch(r"\w+", toks[-1]):
            toks = toks[:-1]                                # drop the name
        out.append(" ".join(toks))
    return tuple(out)


def _declarations(text):
    found = {}
    for ret, stars, name, params in _DECL.findall(_strip_comments(text)):
        toks = _tokens(ret) + _tokens(stars)
        if not toks or toks[0] in _NOT_A_TYPE:
            continue
        found[name] = (" ".join(toks), _params(params))
    return found


@pytest.fixture(scope="module")
def c_declarations():
    found = {}
    for header in _HEADERS:
        found.update(_declarations(header.read_text()))
    return found


@pytest.fixture(scope="module")
def cdef_declarations():
    from geozl import _ffi
    return _declarations(_ffi._CDEF)


def test_the_cdef_is_not_empty(cdef_declarations, c_declarations):
    """A parser that quietly matched nothing would let this whole file pass."""
    assert len(cdef_declarations) > 30
    assert len(c_declarations) > len(cdef_declarations)


def test_every_declared_symbol_exists_in_a_header(cdef_declarations,
                                                  c_declarations):
    missing = sorted(set(cdef_declarations) - set(c_declarations))
    assert not missing, f"declared in _ffi.py but in no header: {missing}"


def test_every_signature_matches_the_header(cdef_declarations,
                                            c_declarations):
    drift = []
    for name, (ret, params) in sorted(cdef_declarations.items()):
        if name not in c_declarations:
            continue
        c_ret, c_params = c_declarations[name]
        if (ret, params) != (c_ret, c_params):
            drift.append(f"{name}\n    C  {c_ret} {c_params}\n"
                         f"    py {ret} {params}")
    assert not drift, "cdef drifted from the headers:\n" + "\n".join(drift)


def test_public_c_api_keeps_the_0_13_signatures():
    baseline = _declarations(_PUBLIC_API_0_13)
    current_api = _declarations(_HEADERS[0].read_text())
    assert len(baseline) == 26
    drift = []
    for name, signature in baseline.items():
        current = current_api.get(name)
        if current != signature:
            drift.append(f"{name}: expected {signature}, found {current}")
    assert not drift, "public C API changed:\n" + "\n".join(drift)


def test_the_parser_sees_a_planted_change(c_declarations):
    """The check above only means something if a real edit trips it."""
    planted = _declarations(
        "int planar_encode(void* dst, const void* src, uint32_t width, "
        "size_t nbElts, size_t eltWidth);")
    assert planted["planar_encode"] != c_declarations["planar_encode"]
