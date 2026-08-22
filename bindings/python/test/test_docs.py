"""Checks for the hand-written documentation, which has no build step."""

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
API_HIGH = DOCS / "api-high.html"
NOTEBOOK = DOCS / "notebooks" / "high-level-api.ipynb"
HIGH_LEVEL_SOURCES = [
    ROOT / "bindings" / "python" / "geozl" / "_2d.py",
    ROOT / "bindings" / "python" / "geozl" / "_coeffs.py",
]


def _maintained_text() -> list[Path]:
    paths = [ROOT / "README.md", ROOT / "bindings" / "python" / "README.md"]
    paths.extend(
        path for path in DOCS.rglob("*")
        if path.is_file() and path.suffix in {".css", ".html", ".ipynb", ".js", ".md"}
        and path.name != "api-low.html"
    )
    return paths


def test_maintained_docs_do_not_reference_the_legacy_low_level_page():
    pattern = re.compile(r"api[-_]low(?:\.html)?", re.I)
    found = []
    for path in _maintained_text():
        if pattern.search(path.read_text(encoding="utf-8")):
            found.append(path.relative_to(ROOT).as_posix())
    assert not found, f"legacy low-level API page referenced by: {', '.join(found)}"


def test_current_docs_do_not_use_the_removed_max_error_argument():
    found = []
    for path in _maintained_text():
        if re.search(r"\bmax_error\b", path.read_text(encoding="utf-8")):
            found.append(path.relative_to(ROOT).as_posix())
    assert not found, f"obsolete max_error argument in: {', '.join(found)}"


def _source_signatures() -> dict[str, str]:
    out = {}
    for source in HIGH_LEVEL_SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in {
                    "profile", "graph", "compress", "decompress", "coeffs"}:
                continue
            positional = node.args.posonlyargs + node.args.args
            defaults = [None] * (len(positional) - len(node.args.defaults)) + node.args.defaults
            parts = []
            for arg, default in zip(positional, defaults, strict=True):
                parts.append(arg.arg if default is None else
                             f"{arg.arg}={_format_default(default)}")
            if node.args.kwonlyargs:
                parts.append("*")
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                parts.append(arg.arg if default is None else
                             f"{arg.arg}={_format_default(default)}")
            assert node.name not in out, f"duplicate public function: {node.name}"
            out[node.name] = f"{node.name}({', '.join(parts)})"
    return out


def _format_default(node: ast.expr) -> str:
    value = ast.literal_eval(node)
    return json.dumps(value) if isinstance(value, str) else repr(value)


def test_api_page_signatures_match_the_public_python_source():
    body = API_HIGH.read_text(encoding="utf-8")
    shown = {
        name: found.group(1)
        for name in ("profile", "graph", "compress", "decompress", "coeffs")
        if (found := re.search(rf"signature <b>({name}\([^<]+\)) →", body))
    }
    assert shown == _source_signatures()


def test_notebook_describes_the_current_four_step_workflow():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ).lower()
    for obsolete in ("three calls", "compress runs the search", "checksum off"):
        assert obsolete not in markdown
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
