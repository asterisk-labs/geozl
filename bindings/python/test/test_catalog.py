"""The docs restate things the source already knows. This checks they still
agree, since a static site with no build step has nothing else that would
notice when they stop.

The codec catalog lives in five places: the CTid in ctids.h, the README table,
the sections of the four family pages, the CODECS array in docs/assets/js/main.js and the
individual codec pages. The four public copies have to match. ctids.h is checked
one way only, since a codec can exist in C before it is documented.

"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

CTIDS = ROOT / "core" / "include" / "geozl" / "ctids.h"
README = ROOT / "README.md"
DOCS = ROOT / "docs"
OVERVIEW = DOCS / "docs.html"
FAMILIES = [DOCS / f"{name}.html" for name in ("predictors", "streams", "fused", "near-lossless")]
PAGER = ROOT / "docs" / "assets" / "js" / "main.js"

pytestmark = pytest.mark.skipif(
    not CTIDS.exists(),
    reason="source tree not next to the package, nothing to cross-check",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ctids() -> dict[str, int]:
    """Name to CTid, from the enum. GEOZL_CTID_DELTA_W becomes delta_w."""
    body = _read(CTIDS)
    out = {}
    for name, value in re.findall(r"GEOZL_CTID_(\w+)\s*=\s*(0x[0-9A-Fa-f]+)", body):
        if name in ("FIRST", "LAST", "LOSSLESS_LAST"):
            continue
        out[name.lower()] = int(value, 16)
    return out


def readme_rows() -> list[tuple[str, int]]:
    """The codec table, as (name, ctid) in the order it is written."""
    body = _read(README)
    rows = re.findall(r"^\|\s*`(\w+)`\s*\|\s*`(0x[0-9A-Fa-f]+)`\s*\|", body, re.M)
    return [(n, int(v, 16)) for n, v in rows]


def catalog_cards(pages: list[Path] = FAMILIES) -> list[dict[str, str]]:
    """The codec sections of the family pages, in reading order."""
    body = "".join(_read(page) for page in pages)
    pattern = (
        r'<section class="codec" id="(\w+)" data-no="(\d+)" '
        r'data-ctid="(0x[0-9A-Fa-f]+)" data-fam="(\w+)" data-page="codecs/([\w.-]+)"'
        r'[^>]*>\s*<h3><span class="num">\2</span>\1</h3>'
    )
    return [
        {"file": f, "fam": fam, "no": no, "ctid": ct, "name": name}
        for name, no, ct, fam, f in re.findall(pattern, body)
    ]


def pager_entries() -> list[dict[str, str]]:
    """The CODECS array in main.js, in array order."""
    body = _read(PAGER)
    pattern = (
        r'\{\s*no:\s*"(\d+)",\s*name:\s*"(\w+)",\s*'
        r'file:\s*"([\w.-]+)",\s*fam:\s*"(\w+)"\s*\}'
    )
    return [
        {"no": no, "name": name, "file": f, "fam": fam}
        for no, name, f, fam in re.findall(pattern, body)
    ]


def codec_page_identity(path: Path) -> dict[str, str]:
    """The identifiers a codec page exposes to readers."""
    body = _read(path)
    patterns = {
        "title": r"<title>GeoZL\s*·\s*(\w+)</title>",
        "name": r'<h1 class="name">(\w+)</h1>',
        "ctid": r"<dt>ctid</dt><dd>(0x[0-9A-Fa-f]+)</dd>",
        "footer_ctid": r'<span class="pg-mid mono">(0x[0-9A-Fa-f]+)</span>',
    }
    out = {}
    for field, pattern in patterns.items():
        found = re.search(pattern, body)
        assert found is not None, f"{path.name} has no parseable {field}"
        out[field] = found.group(1)
    return out


def test_each_source_parses():
    """A rename that breaks the shape above would otherwise pass everything
    below on an empty list."""
    assert len(ctids()) >= 11
    assert len(readme_rows()) >= 11
    assert len(catalog_cards()) >= 11
    assert len(pager_entries()) >= 11


def test_catalog_and_pager_agree():
    cards = catalog_cards()
    entries = pager_entries()
    assert [(c["no"], c["name"], c["fam"]) for c in cards] == [
        (e["no"], e["name"], e["fam"]) for e in entries
    ]
    assert [c["file"] for c in cards] == [e["file"] for e in entries]


def test_readme_table_matches_the_catalog():
    """Same codecs and same ids. Not the same order: the README table reads by
    CTid and the catalog groups by family, and both are right for what they
    are."""
    cards = {c["name"]: int(c["ctid"], 16) for c in catalog_cards()}
    assert dict(readme_rows()) == cards


def test_readme_table_is_sorted_by_ctid():
    ctid_column = [ctid for _, ctid in readme_rows()]
    assert ctid_column == sorted(ctid_column)


def test_documented_ctids_exist_in_the_header():
    """One way. A codec in ctids.h that is not documented yet is allowed, a
    documented codec whose id does not match the header is not."""
    known = ctids()
    for name, ctid in readme_rows():
        assert name in known, f"{name} is in the README with no CTid in ctids.h"
        assert known[name] == ctid, (
            f"{name} is 0x{ctid:X} in the README and 0x{known[name]:X} in ctids.h"
        )


def test_every_codec_page_exists():
    for card in catalog_cards():
        page = DOCS / "codecs" / card["file"]
        assert page.exists(), f"{card['name']} links to a missing {card['file']}"


def test_codec_pages_match_the_catalog():
    for card in catalog_cards():
        expected_file = card["name"].replace("_", "-") + ".html"
        assert card["file"] == expected_file, (
            f"{card['name']} uses unexpected page name {card['file']}"
        )
        page = DOCS / "codecs" / card["file"]
        assert page.exists(), f"{card['name']} links to a missing {card['file']}"
        identity = codec_page_identity(page)
        assert identity["title"] == card["name"]
        assert identity["name"] == card["name"]
        assert int(identity["ctid"], 16) == int(card["ctid"], 16)
        assert int(identity["footer_ctid"], 16) == int(card["ctid"], 16)


def test_catalog_contents_list_every_section():
    for page in FAMILIES:
        listed = re.findall(r'<li><a href="#(\w+)"><span class="toc-no">(\d+)</span>', _read(page))
        assert listed == [(c["name"], c["no"]) for c in catalog_cards([page])]

    total = re.search(r"(\d+) codecs", _read(OVERVIEW))
    assert total is not None
    assert int(total.group(1)) == len(catalog_cards())


def test_no_codec_page_ships_a_dead_link():
    """No codec page links to href="#" or carries a TODO."""
    for page in sorted((DOCS / "codecs").glob("*.html")):
        body = _read(page)
        assert 'href="#"' not in body, f"{page.name} still has a placeholder link"
        assert "TODO" not in body, f"{page.name} still carries a TODO"


def test_code_button_points_at_the_codec_source():
    """The button resolves to core/src/<codec>, which has to be a real folder
    with a binding in it, or the glob in CMakeLists would not call it a codec."""
    for card in catalog_cards():
        page = DOCS / "codecs" / card["file"]
        found = re.search(r'class="btn-code" href="([^"]+)"', _read(page))
        assert found is not None, f"{card['name']} has no Code button"
        name = found.group(1).rsplit("/", 1)[-1]
        assert name == card["name"], f"{card['name']} links to core/src/{name}"
        binding = ROOT / "core" / "src" / name / f"encode_{name}_binding.c"
        assert binding.exists(), f"core/src/{name} is not a codec folder"
