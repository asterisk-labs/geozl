"""The docs restate things the source already knows. This checks they still
agree, since a static site with no build step has nothing else that would
notice when they stop.

The codec catalog lives in five places: the CTid in ctids.h, the README table,
the cards in docs/docs.html, the CODECS array in docs/assets/js/main.js and the
individual codec pages. The four public copies have to match. ctids.h is checked
one way only, since a codec can exist in C before it is documented.

The version lives in VERSION and in the landing badge, which had already
drifted a release behind.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

CTIDS = ROOT / "core" / "include" / "geozl" / "ctids.h"
README = ROOT / "README.md"
CATALOG = ROOT / "docs" / "docs.html"
PAGER = ROOT / "docs" / "assets" / "js" / "main.js"
VERSION = ROOT / "VERSION"
LANDING = ROOT / "docs" / "index.html"

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


def catalog_cards() -> list[dict[str, str]]:
    """The cards on docs.html, in page order."""
    body = _read(CATALOG)
    pattern = (
        r'<a class="card" href="codecs/([\w.-]+)"[^>]*--fam:var\(--(\w+)\)"[^>]*>\s*'
        r'<div class="card-top"><span class="card-no">(\d+)</span>'
        r'<span class="card-ct">(0x[0-9A-Fa-f]+)</span></div>\s*'
        r'<div class="card-name">(\w+)</div>'
    )
    return [
        {"file": f, "fam": fam, "no": no, "ctid": ct, "name": name}
        for f, fam, no, ct, name in re.findall(pattern, body)
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
        "title": r"<title>geozl\s*·\s*(\w+)</title>",
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
        page = CATALOG.parent / "codecs" / card["file"]
        assert page.exists(), f"{card['name']} links to a missing {card['file']}"


def test_codec_pages_match_the_catalog():
    for card in catalog_cards():
        expected_file = card["name"].replace("_", "-") + ".html"
        assert card["file"] == expected_file, (
            f"{card['name']} uses unexpected page name {card['file']}"
        )
        page = CATALOG.parent / "codecs" / card["file"]
        assert page.exists(), f"{card['name']} links to a missing {card['file']}"
        identity = codec_page_identity(page)
        assert identity["title"] == card["name"]
        assert identity["name"] == card["name"]
        assert int(identity["ctid"], 16) == int(card["ctid"], 16)
        assert int(identity["footer_ctid"], 16) == int(card["ctid"], 16)


def test_catalog_legend_counts_the_cards():
    body = _read(CATALOG)
    families = dict(
        (fam, int(n))
        for fam, n in re.findall(
            r'--fam:var\(--(\w+)\)"><span class="lg-sq"></span>[^<]*<b>(\d+)</b>', body
        )
    )
    counted: dict[str, int] = {}
    for card in catalog_cards():
        counted[card["fam"]] = counted.get(card["fam"], 0) + 1
    assert families == counted

    total = re.search(r'<span class="legend-count">(\d+) codecs</span>', body)
    assert total is not None
    assert int(total.group(1)) == len(catalog_cards())


def landing_badge() -> str:
    """The status badge on the landing page, as written."""
    found = re.search(r'<span class="mono status">(.*?)</span>\s*</header>',
                      _read(LANDING), re.S)
    assert found is not None, "no status badge on the landing page"
    return re.sub(r"<[^>]+>", "", found.group(1)).strip()


def test_landing_badge_carries_the_current_version():
    """The badge is the one copy of the version that is not derived from
    VERSION, since the site has no build step to derive it."""
    want = _read(VERSION).strip()
    found = re.match(r"v(\S+)", landing_badge())
    assert found is not None, f"badge {landing_badge()!r} opens with no version"
    assert found.group(1) == want, (
        f"landing page says v{found.group(1)}, VERSION says {want}"
    )


def test_landing_badge_does_not_claim_more_than_the_readme():
    """The site and README should describe the project at the same level."""
    status = "active development"
    assert status in landing_badge().lower()
    assert status in _read(README).lower()


def test_no_codec_page_ships_a_dead_link():
    """Every codec page carried a Code button pointing at href="#", with the
    TODO to fix it still in the markup. Shipped, on all eleven."""
    for page in sorted((CATALOG.parent / "codecs").glob("*.html")):
        body = _read(page)
        assert 'href="#"' not in body, f"{page.name} still has a placeholder link"
        assert "TODO" not in body, f"{page.name} still carries a TODO"


def test_code_button_points_at_the_codec_source():
    """The button resolves to core/src/<codec>, which has to be a real folder
    with a binding in it, or the glob in CMakeLists would not call it a codec."""
    for card in catalog_cards():
        page = CATALOG.parent / "codecs" / card["file"]
        found = re.search(r'class="btn-code" href="([^"]+)"', _read(page))
        assert found is not None, f"{card['name']} has no Code button"
        name = found.group(1).rsplit("/", 1)[-1]
        assert name == card["name"], (
            f"{card['name']} links to core/src/{name}"
        )
        binding = ROOT / "core" / "src" / name / f"encode_{name}_binding.c"
        assert binding.exists(), f"core/src/{name} is not a codec folder"
