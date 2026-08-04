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


def test_the_parser_sees_a_planted_change(c_declarations):
    """The check above only means something if a real edit trips it."""
    planted = _declarations(
        "int planar_encode(void* dst, const void* src, uint32_t width, "
        "size_t nbElts, size_t eltWidth);")
    assert planted["planar_encode"] != c_declarations["planar_encode"]
