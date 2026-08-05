<p align="center">
  <img src="docs/assets/svg/banner.svg" alt="geozl" width="750"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-2b8a3e.svg" alt="License: BSD-3-Clause"/>
  <img src="coverage.svg" alt="Coverage"/>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue" alt="Platform"/>
  <img src="https://img.shields.io/badge/C11-blue" alt="C11"/>
  <a href="https://github.com/facebook/openzl">
    <img src="https://img.shields.io/badge/built%20on-OpenZL-6f42c1" alt="Built on OpenZL"/>
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/linted%20with-ruff-d7ff64" alt="Linted with Ruff"/>
  </a>
  <a href="https://mypy-lang.org/">
    <img src="https://img.shields.io/badge/type%20checked-mypy-1f5082" alt="Type checked with mypy"/>
  </a>
  
</p>

## What is OpenZL and GeoZL?

[OpenZL](https://github.com/facebook/openzl) is a new compression framework that treats compression as a graph of codecs. Each frame carries the recipe needed to decode it, which lets a universal OpenZL decoder follow the graph without knowing how the data was originally encoded.

That model works well for one-dimensional streams, but it does not know that a raster has spatial structure. **GeoZL adds that missing spatial layer.**

A [GeoZL](https://asterisk.coop/geozl/) codec is an OpenZL graph node that understands raster tiles. It transforms a typed numeric stream, stores the metadata needed to reverse that transform in the codec header, and lets the rest of the OpenZL graph continue as usual.

If you want to implement a new codec, see [docs/adding-a-codec.md](docs/adding-a-codec.md).

<p align="center">
  <img src="docs/assets/svg/graph-recipes.svg" alt="geozl" width="750"/>
</p>


## Status

GeoZL is **experimental**.

> [!WARNING]
> **GeoZL codecs are not part of OpenZL.**
>
> They are registered at runtime as OpenZL custom transforms and use CTids in the `0x72D700`-`0x72D7FF` range. A frame that uses GeoZL codecs can only be decoded by a reader that has GeoZL registered. Frames that use only built-in OpenZL codecs remain portable OpenZL frames.

## Install

```bash
pip install geozl
```

## Example

GeoZL has two entry points: a high-level API that compresses a tile in one call, and a low-level API that places individual codecs in an OpenZL graph.

### High-level API

`geozl.profile` measures a set of candidate graphs on your tile and ranks them, `geozl.compress` runs the one you name and returns the frame. It never searches, so the slow call happens once and the fast one happens on every tile after that. `geozl.decompress` reverses the frame back to a tile.

```python
import numpy as np
import geozl

tile = np.random.randint(0, 4096, (1024, 1024), dtype=np.uint16)

rows = geozl.profile(tile)                        # the slow call, run once
best = rows[0]["graph"]                           # e.g. "planar>zigzag>transpose>entropy"

frame = geozl.compress(tile, method=best)         # the fast call, run always
frame = geozl.compress(tile, method=best, error="LINEAR:MAX_ERROR=2")  # near-lossless
frame = geozl.compress(tile, method=best, error="LOG:MAX_ERROR=1%")   # bound follows the value

back = geozl.decompress(frame, dtype="uint16", width=1024)
```

### Low-level API

For anything else, place the codecs in an `openzl.ext` graph yourself, alongside regular OpenZL nodes.

```python
import openzl.ext as zl
import geozl

c = zl.Compressor()
g = zl.graphs.Compress()

g = zl.nodes.Zigzag()(c, g)
g = geozl.lossless.Planar(width=512)(c, g)

c.select_starting_graph(g)
```

### Decoding

Either way, a reader has to register the geozl decoders before it can follow the frame.

```python
import openzl.ext as zl
import geozl

d = zl.DCtx()
geozl.register_decoders(d)
tile = d.decompress(frame)[0].content.as_nparray()
```

## Codecs

| codec | CTid | what it does |
|---|---:|---|
| `delta_w` | `0x72D701` | residual against the west neighbour |
| `delta_n` | `0x72D702` | residual against the north neighbour |
| `planar` | `0x72D703` | predicts each pixel from `W + N - NW` |
| `deinterleave` | `0x72D704` | separates a two-lane interleaved stream |
| `med` | `0x72D705` | median edge detector predictor |
| `average` | `0x72D706` | floor average of the west and north neighbours |
| `wp_static` | `0x72D707` | fits a weighted predictor and stores the weights in the frame |
| `nodata` | `0x72D70C` | pulls missing samples into a validity mask and fills the holes |
| `quant_linear` | `0x72D781` | uniform grid, fixed absolute bound, `LINEAR:MAX_ERROR=V` |
| `quant_log` | `0x72D782` | logarithmic grid, bound is a fraction of the value, `LOG:MAX_ERROR=P%` |
| `quant_sqrt` | `0x72D783` | square root grid, bound grows with the sensor noise, `SQRT:MAX_ERROR=VN` |

## License

BSD-3-Clause

<div align="center">
  <br>
  Made with &#9829; by
  <br><br>
  <a href="https://asterisk.coop">
    <img src="docs/assets/svg/asterisk_banner.svg" alt="Asterisk Labs" width="400"/>
  </a>
</div>
