# geozl — site

Landing and section skeletons for the geozl documentation site.
Light "datasheet" design: bordered panels, Space Grotesk + IBM Plex Mono, no shadows.

## Structure

```
geozl-site/
├── index.html          landing (sheet · spec table · index · distribution)
├── docs.html           Documentation
├── api-high.html       High-level API   — compress()
├── api-low.html        Low-level API    — build the OpenZL graph in Python
├── benchmark.html      Benchmark        — ratio vs speed
└── assets/
    ├── css/style.css
    ├── js/main.js       footer year + active nav
    └── svg/
        ├── mark.svg         animated 2×2 prediction mark
        ├── asterisk.svg     Asterisk Labs mark
        ├── favicon.svg
        ├── github.svg       ┐
        ├── huggingface.svg  ├ distribution logos
        └── source-coop.svg  ┘ (source.coop recolored for light bg)
```

Distribution links (GitHub / Hugging Face / source.coop) use placeholder URLs
for now — swap them for the real ones when the repo and dataset are up.

## Page transitions

Cross-document transitions are CSS-only: `@view-transition { navigation: auto; }`.
The GeoZL lockup carries `view-transition-name: geozl-lockup`, so it morphs from the
landing hero into each page's masthead. Needs http(s) + a supporting browser.

## Run

```bash
cd geozl-site
python3 -m http.server 8000
# http://localhost:8000
```

## Refined layout

The responsive structure keeps the original copy and links unchanged. The desktop layout uses a wider technical sheet, an asymmetric 12-column navigation index, and split inner pages so sparse documentation skeletons still occupy the available canvas. Mobile remains a single-column, touch-friendly layout.
