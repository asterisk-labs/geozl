# The geozl site

Static HTML, no build step and no framework. Every page is hand written, the CSS
is one file plus one page-specific sheet, and the JavaScript is optional
everywhere. Open any file from disk and it works, minus the cross-document
transitions, which need http.

The design is a datasheet. Bordered panels, Space Grotesk for text and IBM Plex
Mono for anything the reader could type, paper `#F1EDE4` on ink `#17161C`, and no
shadows anywhere. Each codec family carries a colour that follows it from the
catalog cards through the pager to the page header, purple for predictors, green
for the two that split a stream, orange for the near lossless three.

## Structure

```
docs/
├── index.html          landing, spec table and distribution
├── docs.html           codec catalog, one card per codec
├── codecs/*.html       11 pages, one per codec, each with an animated diagram
├── api-high.html       graph(), compress(), profile() and the lossy recipes
├── benchmark.html      ratio against throughput
├── links.html          GitHub, Hugging Face, source.coop
├── c-api.md            C source compatibility policy
├── adding-a-codec.md   how to write one, for contributors
├── notebooks/          high-level-api.ipynb, the runnable version of api-high
└── assets/
    ├── css/style.css       everything shared
    ├── css/api-high.css     only api-high.html, which is the heaviest page
    ├── js/main.js           codec pager, loaded by codecs/*.html
    ├── js/deck.js           deck navigation on api-high
    ├── js/tabs.js           the function picker on api-high
    ├── js/copy.js           copy buttons over the code blocks
    ├── js/plot-tooltip.js   nearest-point tooltip for any .plot
    ├── js/dem-readout.js    elevation readout over the hero tile
    ├── img/                 the Mont Blanc tile and the three loop frames
    └── svg/
        ├── mark.svg          animated 2x2 prediction mark
        ├── banner.svg        the README banner
        ├── codecs/*.svg      11 diagrams, one per codec page
        ├── asterisk.svg, asterisk_banner.svg
        ├── github.svg, huggingface.svg, source-coop.svg
        └── favicon.svg, heart.svg, check.svg, copy.svg, colab.svg, chevron-*.svg
```

## The catalog is written down five times

`core/include/geozl/ctids.h` holds the CTid, the README table lists the codecs, the
cards in `docs.html` show them, and the `CODECS` array in `main.js` walks them for
prev and next. Each codec page repeats its name and CTid. Adding a codec means
updating all five representations, and they used to drift without anything
noticing.

`bindings/python/test/test_catalog.py` cross-checks them, so a codec renamed in
one place fails `make test`. The CTid header is checked one way only, since a
codec can exist in C before it is documented.

## Page transitions

Cross-document transitions are CSS only, `@view-transition { navigation: auto; }`
in `style.css`. The lockup carries `view-transition-name: geozl-lockup`, so it
morphs from the landing hero into each page's masthead. Needs http and a browser
that supports it, and degrades to a plain navigation everywhere else.

## The active nav link is in the HTML

Each page marks its own nav entry with `class="here"`. There is no runtime pass
for it, which is why the pages that carry no scripts at all still highlight
correctly.

## Run

```bash
python3 -m http.server 8000 --directory docs
# http://localhost:8000
```
