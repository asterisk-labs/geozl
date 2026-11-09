"""Checks for the directly maintained documentation, which has no build step."""

import html
import importlib
import inspect
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
API = DOCS / "api.html"
NOTEBOOK = DOCS / "notebooks" / "high-level-api.ipynb"
SKILL = ROOT / ".claude" / "skills" / "geozl"
CODEX_SKILL = ROOT / ".agents" / "skills" / "geozl"
PUBLIC_HEADERS = [
    ROOT / "core" / "include" / "geozl" / name
    for name in ("geozl.h", "coeffs.h", "gpu.h")
]


def test_agent_skill_tracks_the_release_and_its_references_resolve():
    entrypoint = SKILL / "SKILL.md"
    body = entrypoint.read_text(encoding="utf-8")
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

    assert f"This skill describes **geozl {version}**" in body
    assert CODEX_SKILL.resolve() == SKILL.resolve()

    failures = []
    for source in SKILL.rglob("*.md"):
        text = source.read_text(encoding="utf-8")
        for relative in re.findall(r"\[[^]]*\]\((?!https?://)([^)#]+\.md)(?:#[^)]*)?\)", text):
            if not (source.parent / relative).is_file():
                failures.append(f"{source.relative_to(SKILL)}: {relative}")

    assert not failures, "broken skill references:\n" + "\n".join(failures)


def _sig_lines(body: str, kind: str) -> list[str]:
    blocks = re.findall(rf'<pre><code class="{kind}">(.*?)</code></pre>', body, re.S)
    return [html.unescape(block) for block in blocks]


def _plain_signature(obj: object, *, drop_self: bool) -> str:
    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    if drop_self:
        params = params[1:]
    params = [p.replace(annotation=inspect.Parameter.empty) for p in params]
    text = str(sig.replace(parameters=params,
                           return_annotation=inspect.Signature.empty))
    # a constructor's "-> None" says nothing about the class
    if not inspect.isclass(obj) and sig.return_annotation is not inspect.Signature.empty:
        returns = inspect.formatannotation(sig.return_annotation)
        text += " -> " + re.sub(r"\b(?:\w+\.)+(\w+)", r"\1", returns)
    return text


def _resolve(dotted: str) -> tuple[object, bool]:
    parts = dotted.split(".")
    assert parts[0] == "geozl", dotted
    owner, obj = None, importlib.import_module("geozl")
    for part in parts[1:]:
        owner, obj = obj, getattr(obj, part)
    return obj, inspect.isclass(owner) and inspect.isfunction(obj)


def test_api_page_python_signatures_match_the_code():
    pytest.importorskip("openzl")
    shown = []
    for block in _sig_lines(API.read_text(encoding="utf-8"), "sig"):
        for line in block.splitlines():
            if line[:1].isspace():  # a long signature wraps onto indented lines
                shown[-1] += " " + line.strip()
            elif line.strip():
                shown.append(line)
    assert shown, "api.html shows no Python signatures"

    failures = []
    names = set()
    for line in shown:
        dotted = line.split("(", 1)[0]
        names.add(dotted)
        obj, drop_self = _resolve(dotted)
        real = dotted + _plain_signature(obj, drop_self=drop_self)
        if line != real:
            failures.append(f"page {line}\ncode {real}")
    assert not failures, "signatures differ:\n" + "\n".join(failures)

    geozl = importlib.import_module("geozl")
    expected = {f"geozl.{name}" for name in geozl.__all__
                if callable(getattr(geozl, name)) and not inspect.ismodule(getattr(geozl, name))}
    for module in (geozl.lossless, geozl.lossy):
        expected |= {f"{module.__name__}.{name}" for name in module.__all__
                     if not name.endswith("Decoder")}
    assert expected <= names, f"undocumented: {sorted(expected - names)}"


def _normalize_c(decl: str) -> str:
    decl = re.sub(r"\s+", " ", decl).strip()
    return re.sub(r"\s*([(),*])\s*", r"\1", decl).rstrip(";")


def test_api_page_c_prototypes_match_the_public_headers():
    declared = set()
    for header in PUBLIC_HEADERS:
        text = re.sub(r"//[^\n]*", "", header.read_text(encoding="utf-8"))
        for decl in re.findall(r"GEOZL_API\s+([^;{]+;)", text):
            declared.add(_normalize_c(decl))

    shown = set()
    for block in _sig_lines(API.read_text(encoding="utf-8"), "proto"):
        for decl in block.split(";"):
            if decl.strip():
                shown.add(_normalize_c(decl))

    assert shown == declared, (
        f"missing from api.html: {sorted(declared - shown)}\n"
        f"not in the headers: {sorted(shown - declared)}")


def test_notebook_describes_the_current_four_step_workflow():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ).lower()
    for call in ("profile", "graph", "compress", "decompress"):
        assert f"`{call}`" in markdown


class _HTMLRefs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "a" and values.get("name"):
            self.ids.add(values["name"])
        for attr in ("href", "src"):
            if values.get(attr) is not None:
                self.refs.append(values[attr] or "")
        if tag == "object" and values.get("data") is not None:
            self.refs.append(values["data"] or "")


def _parse_html(path: Path) -> _HTMLRefs:
    parsed = _HTMLRefs()
    parsed.feed(path.read_text(encoding="utf-8"))
    return parsed


def test_html_local_links_assets_and_fragments_resolve():
    pages = sorted(DOCS.glob("*.html")) + sorted((DOCS / "codecs").glob("*.html"))
    parsed = {page.resolve(): _parse_html(page) for page in pages}
    failures = []

    for page in pages:
        for link in parsed[page.resolve()].refs:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue

            raw_path = unquote(parts.path)
            if raw_path.startswith("/"):
                target = DOCS / raw_path.lstrip("/")
            elif raw_path:
                target = page.parent / raw_path
            else:
                target = page
            target = target.resolve()

            if not target.exists():
                failures.append(f"{page.relative_to(DOCS)}: {link!r} -> missing {target}")
                continue
            if parts.fragment and target.suffix.lower() == ".html":
                target_doc = parsed.get(target)
                if target_doc is None:
                    target_doc = _parse_html(target)
                    parsed[target] = target_doc
                fragment = unquote(parts.fragment)
                if fragment not in target_doc.ids:
                    failures.append(
                        f"{page.relative_to(DOCS)}: {link!r} -> "
                        f"missing fragment #{fragment} in {target.relative_to(DOCS)}"
                    )

    assert not failures, "broken local documentation references:\n" + "\n".join(failures)
