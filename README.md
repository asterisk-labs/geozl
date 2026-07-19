<p align="center">
  <img src="img/geozl-banner-dark.svg" alt="geozl" width="750"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-2b8a3e.svg" alt="License: BSD-3-Clause"/>
  <img src="coverage.svg" alt="Coverage"/>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue" alt="Platform"/>
  <img src="https://img.shields.io/badge/C11-blue" alt="C11"/>
  <a href="https://github.com/facebook/openzl">
    <img src="https://img.shields.io/badge/built%20on-OpenZL-6f42c1" alt="Built on OpenZL"/>
  </a>
</p>

## What is OpenZL and GeoZL?

[OpenZL](https://github.com/facebook/openzl) is a new compression framework that treats compression as a graph of codecs. Each frame carries the recipe needed to decode it, which lets a universal OpenZL decoder follow the graph without knowing how the data was originally encoded.

That model works well for one-dimensional streams, but it does not know that a raster has rows, columns, neighbours, or spatial structure. **geozl adds that missing spatial layer.**

A geozl codec is an OpenZL graph node that understands raster tiles. It transforms a typed numeric stream, stores the metadata needed to reverse that transform in the codec header, and lets the rest of the OpenZL graph continue as usual.

If you want to implement a new codec, see [docs/adding-a-codec.md](docs/adding-a-codec.md).

## Status

geozl is **experimental**.

> [!WARNING]
> **geozl codecs are not part of OpenZL.**
>
> They are registered at runtime as OpenZL custom transforms and use CTids in the `0x72D700`-`0x72D7FF` range. A frame that uses geozl codecs can only be decoded by a reader that has geozl registered. Frames that use only built-in OpenZL codecs remain portable OpenZL frames.

## Install

```bash
pip install geozl
```

## Example

geozl has two entry points: a high-level API that compresses a tile in one call, and a low-level API that places individual codecs in an OpenZL graph.

### High-level API

`geozl.compress` takes a tile and returns a frame. With the default method it sweeps every predictor and keeps the smallest one.

```python
import numpy as np
import geozl

tile = np.random.randint(0, 4096, (1024, 1024), dtype=np.uint16)

frame = geozl.compress(tile)                 # sweep, keep the smallest frame
frame = geozl.compress(tile, method="med")   # or name one predictor
frame = geozl.compress(tile, max_error=2)    # near-lossless, absolute bound
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

geozl currently provides two codec families:

- **near-lossless codecs**, under `geozl.lossy`
- **lossless codecs**, under `geozl.lossless`

Both families are registered as OpenZL custom codecs and can be chained with other OpenZL graph nodes.

The `call` column shows the Python call used to place the codec in a graph.

### Near-lossless codecs

Near-lossless codecs quantize the tile once, then store enough information in the frame to report and bound the reconstruction error. A near-lossless frame is no longer bit-exact, it declares the bound it holds to instead.

A frame carries at most one near-lossless codec, as the head transform, so the loss happens once and every stage after it is lossless.

| codec | call | CTid | error |
|---|---|---:|---|
| `quant_linear` | `geozl.lossy.QuantLinear(max_error, dtype)` | `0x72D780` | every value reconstructs within `max_error`, an absolute tolerance |

### Lossless codecs

Lossless codecs are bit-exact transforms over a raster tile. After decoding, the reconstructed tile is identical to the original input.

**Predictors** replace each sample with its residual against a prediction from neighbours the decoder already holds. They take the row width, and one successor.

| codec | call | CTid | what it does |
|---|---|---:|---|
| `delta_w` | `geozl.lossless.DeltaW(width)` | `0x72D701` | stores each value as a difference from its west neighbour |
| `delta_n` | `geozl.lossless.DeltaN(width)` | `0x72D702` | stores each value as a difference from its north neighbour |
| `planar` | `geozl.lossless.Planar(width)` | `0x72D703` | predicts each pixel from `W + N - NW` |
| `med` | `geozl.lossless.Med(width)` | `0x72D705` | uses the median edge detector predictor |
| `average` | `geozl.lossless.Average(width)` | `0x72D706` | predicts from the floor average of west and north neighbours |
| `wp_static` | `geozl.lossless.WpStatic(width)` | `0x72D707` | fits a static weighted predictor and stores the weights in the frame |

**Splits** cut one stream into two lanes, so they take two successors, one per lane, and each lane can go on to its own graph.

| codec | call | CTid | lanes | what it does |
|---|---|---:|---|---|
| `deinterleave` | `geozl.lossless.Deinterleave()` | `0x72D704` | both to one successor | separates a two-lane interleaved stream; for complex, view the tile through `geozl.lossless.component_dtype` first, OpenZL has no complex type |

## License

BSD-3-Clause

<div align="center">
  <br>
  Made with &#9829; by
  <br><br>
  <a href="https://asterisk.coop">
    <img src="img/asterisk_banner.svg" alt="Asterisk Labs" width="400"/>
  </a>
</div>
