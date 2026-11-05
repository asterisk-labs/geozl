# The GeoZL site

The site is maintained directly as static HTML, CSS and JavaScript in `docs/`.
There is no build step or framework. The pages open from disk; cross-document
view transitions need HTTP and a browser that supports them. JavaScript adds
interactions such as copy buttons, tab switching and slide navigation.

The current design uses a dark background, light text and colour accents for
codec families. The landing page keeps its styles and scripts in `index.html`.
The other pages share `style.css` and `dark.css`; the API walkthrough also loads
`api-high.css`.

## Structure

```
docs/
├── index.html          landing page, with its own styles and scripts
├── docs.html           codec catalog, one card per codec
├── codecs/*.html       17 pages, one per documented codec
├── api-high.html       graph(), compress(), profile() and the lossy recipes
├── benchmark.html      benchmark overview and link to the API walkthrough
├── compatibility.md    frame compatibility policy
├── c-api.md            C source compatibility policy
├── releasing.md        release gates and version procedure
├── adding-a-codec.md   how to write one, for contributors
├── notebooks/          high-level-api.ipynb, the runnable version of api-high
├── slides/cng-2026/    conference presentation, with its own assets
└── assets/
    ├── css/style.css       base styles for pages other than the landing page
    ├── css/dark.css        dark theme for those pages
    ├── css/api-high.css    API walkthrough styles
    ├── js/main.js           codec list read by test_catalog.py; no page loads it
    ├── js/deck.js           horizontal slide navigation on api-high
    ├── js/tabs.js           the function picker on api-high
    ├── js/copy.js           copy buttons over the code blocks
    ├── js/plot-tooltip.js   nearest-point tooltip for any .plot
    ├── js/dem-readout.js    elevation readout over the hero tile
    ├── img/                 the DEM tile and the three loop frames
    └── svg/
        ├── mark.svg          animated 2x2 prediction mark
        ├── banner.svg        the README banner
        ├── codecs/*.svg      17 diagrams, one per codec page
        ├── asterisk-mark.svg, asterisk_banner.svg
        ├── github.svg, huggingface.svg, source-coop.svg
        └── favicon.svg, heart.svg, colab.svg, chevron-*.svg
```

## The catalog is written down five times

`core/include/geozl/ctids.h` holds the CTid, the README table lists the codecs, the
cards in `docs.html` show them, and the `CODECS` array in `main.js` lists them for
`test_catalog.py`. Each codec page repeats its name and CTid and carries its prev
and next links as static HTML. Adding a codec means updating all five
representations.

`bindings/python/test/test_catalog.py` cross-checks them, so a codec renamed in
one place fails `make test`. The CTid header is checked one way only, since a
codec can exist in C before it is documented.

## Page transitions

Cross-document transitions are CSS only, `@view-transition { navigation: auto; }`
in `style.css`. The lockup carries `view-transition-name: geozl-lockup`, so it
morphs from the landing hero into each page's masthead. Needs http and a browser
that supports it, and degrades to a plain navigation everywhere else.

## The active nav link is in the HTML

The catalog, API, benchmark and codec pages mark their own nav entry with
`class="here"`. There is no runtime pass for it, so pages without scripts still
highlight the current section.

## Credits

`assets/svg/heart.svg` is the `favorite` glyph from Material Icons, Apache-2.0.

## Run

```bash
python3 -m http.server 8000 --directory docs
# http://localhost:8000
```
