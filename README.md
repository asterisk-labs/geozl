<div align="center">
  <img src="docs/assets/svg/banner.svg" alt="GeoZL" width="750"/>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-2b8a3e?style=flat-square" alt="License BSD-3-Clause"/></a>
    <a href="https://github.com/asterisk-labs/geozl/actions/workflows/ci.yml"><img src="https://github.com/asterisk-labs/geozl/actions/workflows/ci.yml/badge.svg?event=push" alt="CI"/></a>
    <img src="coverage.svg" alt="Coverage"/>
    <a href="https://pypi.org/project/geozl"><img src="https://img.shields.io/pypi/v/geozl?label=python&logo=python&logoColor=white&color=3776AB&style=flat-square" alt="Python"/></a>
    <a href="bindings/r/README.md"><img src="https://img.shields.io/badge/R-geozl-276DC3?logo=r&logoColor=white&style=flat-square" alt="R"/></a>
    <a href="bindings/julia/README.md"><img src="https://img.shields.io/badge/julia-GeoZL.jl-9558B2?logo=julia&logoColor=white&style=flat-square" alt="Julia"/></a>
    <a href="tools/wasm/README.md"><img src="https://img.shields.io/badge/wasm64-JavaScript-654FF0?logo=webassembly&logoColor=white&style=flat-square" alt="WebAssembly"/></a>
    <a href="https://github.com/asterisk-labs/geozl/actions/workflows/platforms.yml"><img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-0078D6?style=flat-square" alt="Linux, macOS and Windows"/></a>
    <a href="https://github.com/facebook/openzl"><img src="https://img.shields.io/badge/built%20on-OpenZL-6f42c1?style=flat-square" alt="Built on OpenZL"/></a>
  </p>
</div>

---

GeoZL compresses raster tiles. [OpenZL](https://github.com/facebook/openzl)
represents compression as a graph of codecs, and GeoZL adds the nodes Earth
observation data needs: spatial predictors, NoData masks, a block bit packer and
bounded-error quantizers.

Frames written by 0.14.0 are the compatibility baseline, and every binding reads
and writes the same frames. See [compatibility](docs/compatibility.md).

## Bindings

| Language | Install | Role | Docs |
|----------|---------|------|------|
| Python | `pip install geozl` | full API and OpenZL nodes | [guide](docs/api-high.html) |
| R | `remotes::install_github("asterisk-labs/geozl", subdir = "bindings/r")` | graph, compress, decompress, profile | [README](bindings/r/README.md) |
| Julia | `Pkg.add(url = "https://github.com/asterisk-labs/geozl", subdir = "bindings/julia")` | graph, compress, decompress, profile | [README](bindings/julia/README.md) |
| JavaScript | `npm install @asterisk-labs/geozl` | compress, decompress | [README](tools/wasm/README.md) |
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

`error=2` bounds the absolute error, `error="1%"` the relative one, and
`nodata=-9999` masks a sentinel. The [changelog](CHANGELOG.md) records each
release.

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
