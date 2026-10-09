<div align="center">
  <img src="docs/assets/svg/banner.svg" alt="GeoZL" width="750"/>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-2b8a3e?style=flat-square" alt="License BSD-3-Clause"/></a>
    <a href="https://github.com/asterisk-labs/geozl/actions/workflows/ci.yml"><img src="https://github.com/asterisk-labs/geozl/actions/workflows/ci.yml/badge.svg?event=push" alt="CI"/></a>
    <img src="coverage.svg" alt="Coverage"/>
    <a href="https://pypi.org/project/geozl"><img src="https://img.shields.io/pypi/v/geozl?label=python&logo=python&logoColor=white&color=3776AB&style=flat-square" alt="Python"/></a>
    <a href="https://asterisk-labs.r-universe.dev/geozl"><img src="https://asterisk-labs.r-universe.dev/badges/geozl" alt="R-universe"/></a>
    <a href="bindings/julia/README.md"><img src="https://img.shields.io/badge/julia-GeoZL.jl-9558B2?logo=julia&logoColor=white&style=flat-square" alt="Julia"/></a>
    <a href="https://www.npmjs.com/package/@asterisk-labs/geozl"><img src="https://img.shields.io/npm/v/%40asterisk-labs%2Fgeozl?label=npm&logo=npm&style=flat-square" alt="npm"/></a>
    <a href="https://github.com/asterisk-labs/geozl/actions/workflows/platforms.yml"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-0078D6?style=flat-square" alt="Linux, macOS and Windows"/></a>
    <a href="https://github.com/facebook/openzl"><img src="https://img.shields.io/badge/built%20on-OpenZL-6f42c1?style=flat-square" alt="Built on OpenZL"/></a>
  </p>
</div>

---

GeoZL extends [OpenZL](https://github.com/facebook/openzl) with codecs designed
for Earth observation data. It adds spatial prediction, NoData
handling, efficient integer packing and bounded-error quantization for
multidimensional numeric arrays. The same core is available from C, Python, R,
Julia and JavaScript.

## Bindings

| Language | Install | Role | Docs |
|----------|---------|------|------|
| Python | `pip install geozl` | full API and OpenZL nodes | [guide](docs/api-high.html) |
| R | `install.packages("geozl", repos="https://asterisk-labs.r-universe.dev")` | graph, compress, decompress, profile | [R-universe](https://asterisk-labs.r-universe.dev/geozl) · [README](bindings/r/README.md) |
| Julia | `Pkg.add(url="https://github.com/asterisk-labs/geozl", subdir="bindings/julia")` | graph, compress, decompress, profile | [GeoZL.jl](bindings/julia/README.md) |
| JavaScript | `npm install @asterisk-labs/geozl` | compress, decompress | [npm](https://www.npmjs.com/package/@asterisk-labs/geozl) · [README](tools/wasm/README.md) |
| C | `make install` | the core | [C API](docs/c-api.md) |

## Quick start

```python
import numpy as np
import geozl

y, x = np.mgrid[0:1024, 0:1024]
tile = (2000 + 8 * y + 5 * x).astype(np.uint16)

best = geozl.profile(tile)[0]["graph"]        # rank recipes on a sample
g = geozl.graph(tile, best)                   # build once, reuse per tile
frame = geozl.compress(tile, graph=g)
back = geozl.decompress(frame).view(np.uint16).reshape(tile.shape)
```

## Codecs

| codec | CTid | what it does |
|-------|-----:|--------------|
| `delta_w` | `0x72D701` | residual against the west neighbour |
| `delta_n` | `0x72D702` | residual against the north neighbour |
| `planar` | `0x72D703` | predicts each pixel from `W + N - NW` |
| `deinterleave` | `0x72D704` | separates a two-lane interleaved stream |
| `med` | `0x72D705` | median edge detector predictor |
| `average` | `0x72D706` | floor average of the west and north neighbours |
| `wp_static` | `0x72D707` | weighted predictor with its weights in the frame |
| `nodata` | `0x72D70C` | missing samples to a validity mask, holes filled |
| `pfor` | `0x72D70D` | bit packs blocks of 256 and patches overflows |
| `planar_zigzag` | `0x72D70F` | planar residuals and Zigzag in one pass |
| `planar_zigzag_pfor` | `0x72D710` | planar, Zigzag and PFOR as one codec |
| `med_zigzag` | `0x72D711` | MED residuals and Zigzag in one pass |
| `planar_zigzag_pivco` | `0x72D712` | planar and Zigzag, then PivCo or PFOR per byte lane |
| `quant_linear` | `0x72D781` | absolute bound: `LINEAR:MAX_ERROR=V` |
| `quant_log` | `0x72D782` | relative bound: `LOG:MAX_ERROR=P%` |
| `quant_sqrt` | `0x72D783` | noise-scaled bound: `SQRT:MAX_ERROR=VN` |

The [codec catalog](docs/docs.html) documents each wire format.

## AI agent skill

Install the [GeoZL skill](https://github.com/asterisk-labs/geozl/blob/main/.claude/skills/geozl/SKILL.md) so coding agents know its graph recipes, bounded-error rules and codec internals.

```bash
npx skills add asterisk-labs/geozl
```

## Development

```bash
make submodules
make test       # C tests and pytest
make help       # R, Julia, wasm, sanitizers and fuzzing
```

## License

BSD-3-Clause.

<div align="center">
  <br>
  Made with &#9829; by
  <br><br>
  <a href="https://asterisk.coop">
    <img src="docs/assets/svg/asterisk_banner.svg" alt="Asterisk Labs" width="400"/>
  </a>
</div>
