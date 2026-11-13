# The GeoZL site

The site is maintained as static HTML, CSS and JavaScript in `docs/`. There is
no framework or deployment build. The notebook pages are generated from their
`.ipynb` sources with `tools/render_notebooks.py` and committed with them. Most
pages open from disk; cross-document view transitions need HTTP and a browser
that supports them. JavaScript adds the section highlight, copy buttons and
figure steppers.

The landing page and codec catalog use a dark background. The reading, API and
notebook pages use a light background. The landing page keeps its styles and
scripts in `index.html`; the other pages share `style.css`, `dark.css` and
`reading.css` as needed.

## Structure

```
docs/
├── index.html          landing page, with its own styles and scripts
├── docs.html           codecs overview and the reading order
├── predictors.html, streams.html, fused.html, near-lossless.html  the four codec families
├── pfor.html           how PFOR works
├── pivco.html          how PivCo-Huffman works
├── cpu.html            how the CPU decoders use vector lanes
├── gpu-codecs.html     why PFOR and PivCo decode well on a GPU
├── gpu-internals.html  how the GPU decoder is built
├── slang.html          the GPU kernels, written once in Slang
├── api.html            API overview, the four calls and one line per language
├── api-python.html     every public Python function, checked by test_docs.py
├── api-c.html          every public C function, checked by test_docs.py
├── api-gpu.html        the geozl/gpu.h API, as a guide
├── api-r.html, api-julia.html, api-js.html  the R, Julia and JavaScript bindings
├── examples.html       example notebooks that open in Colab, and the ones coming soon
├── codecs/*.html       17 pages, one per documented codec
├── compatibility.md    frame compatibility policy
├── c-api.md            C source compatibility policy
├── releasing.md        release gates and version procedure
├── adding-a-codec.md   how to write one, for contributors
├── notebooks/          five notebook sources, their generated HTML pages and figures
├── slides/cng-2026/    conference presentation, with its own assets
└── assets/
    ├── css/style.css       base styles for pages other than the landing page
    ├── css/dark.css        dark theme for those pages
    ├── css/reading.css     shared layout of the reading pages and the API pages
    ├── css/notebook.css    notebook cells, outputs and figures
    ├── js/main.js           codec list read by test_catalog.py; no page loads it
    ├── js/toc.js            highlights the current section in the docs sidebar
    ├── js/stepper.js        steps through the figures marked .stepper
    ├── js/code.js           colors the code blocks and adds their copy buttons
    └── svg/
        ├── mark.svg          animated 2x2 prediction mark
        ├── banner.svg        the README banner
        ├── codecs/*.svg      17 diagrams, one per codec page
        ├── codecs-light/*.svg  the same diagrams in light colors, for the reading pages
        ├── cpu/*.svg         figures of cpu.html
        ├── gpu/*.svg         figures of the PFOR, PivCo, GPU and API GPU pages
        ├── asterisk_banner.svg
        ├── partners/*.svg    Asterisk, Earth Compress, NOC and ARIA logos for the footer and README
        ├── huggingface.svg, source-coop.svg
        └── favicon.svg, heart.svg
```

## The catalog is written down five times

`core/include/geozl/ctids.h` holds the CTid, the README table lists the codecs, the
sections of the four family pages describe them, and the `CODECS` array in `main.js` lists them for
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

Each page marks its own entry, in the header or the sidebar, with
`class="here"`. There is no runtime pass for it, so pages without scripts still
highlight the current section.

## Credits

`assets/svg/heart.svg` is the `favorite` glyph from Material Icons, Apache-2.0.

## Run

```bash
python3 -m http.server 8000 --directory docs
# http://localhost:8000
```
