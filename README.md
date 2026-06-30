<p align="center">
  <img src="img/geozl-banner-dark.svg" alt="geozl" width="750"/>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-2b8a3e.svg" alt="License: BSD-3-Clause"/>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue" alt="Platform"/>
  <img src="https://img.shields.io/badge/C11-blue" alt="C11"/>
  <a href="https://github.com/facebook/openzl"><img src="https://img.shields.io/badge/built%20on-OpenZL-6f42c1" alt="Built on OpenZL"/></a>
</p>
<p align="center"><i>Spatial predictors and near lossless quantizers, packaged as OpenZL custom codecs.</i></p>

OpenZL models compression as a graph of codecs and ships the decode recipe inside every frame, so one **universal decoder** reads any frame. It compresses 1D streams and has no notion of a 2D raster. geozl adds the spatial codecs for it.

A geozl codec is one node in the [OpenZL](https://github.com/facebook/openzl) graph. It transforms the tile, a numeric stream, and carries whatever the decoder needs to invert it in the codec header inside the frame. The full rules live in the [specification](SPEC.md).

## Status

geozl is **experimental**. The codecs, their parameters, and the codec header layout can change between versions with no migration path. Pin a version for anything you cannot regenerate.

> [!WARNING]
> **geozl codecs are not part of OpenZL.** They register at runtime and take custom transform ids in the `0x72D700`-`0x72D7FF` block. A frame that uses them decodes only with a reader that has geozl registered, a frame of built-in OpenZL codecs stays portable. Whether any geozl codec lands in OpenZL upstream is undecided.

## Install

```bash
pip install geozl
```

You compose the geozl codecs into an openzl.ext graph, they chain like any other node.

```python
import openzl.ext as zl
import geozl

c = zl.Compressor()
g = zl.graphs.Compress()
g = zl.nodes.Zigzag()(c, g)
g = geozl.lossless.PlanarInt(width=512)(c, g)
c.select_starting_graph(g)
```

## Codecs

Two families, both registered as OpenZL custom codecs that chain like any other node.

### Near lossless

Quantizers, namespace `geozl.lossy`. A frame is no longer bit exact, carries one quantizer at the head, and bounds its error by the parameters in the frame. Each declares one error mode. **ABS** holds a fixed absolute tolerance, the same everywhere, for elevation, depth, coordinates. **REL** holds a fixed relative tolerance, a percentage of each value, for radiance, reflectance, SAR amplitude.

| codec | CTid | mode | error |
|---|---|---|---|
| `quant_linear` | `0x72D780` | ABS | every value within `max_error` |
| `quant_log` | `0x72D781` | REL | every value within `rel_error` of itself |

### Lossless

Spatial predictors, namespace `geozl.lossless`. Each rewrites a tile as residuals against decoded neighbours, bit exact. They are reversible and domain agnostic, so these are the candidates to merge into OpenZL upstream.

| codec | CTid | what it does |
|---|---|---|
| `delta_w` | `0x72D701` | horizontal delta |
| `delta_n` | `0x72D702` | vertical delta |
| `planar` | `0x72D703` | predicts each pixel from W plus N minus NW |

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