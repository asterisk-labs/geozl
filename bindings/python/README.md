# geozl

Python bindings to [libgeozl](https://github.com/asterisk-labs/geozl), the
geospatial codec extensions for [OpenZL](https://github.com/facebook/openzl).

```python
import numpy as np
import geozl

tile = np.fromfile("b04.raw", dtype=np.uint16).reshape(1024, 1024)

g = geozl.graph(tile, "planar>zigzag>transpose>entropy")
frame = geozl.compress(tile, graph=g)
back = geozl.decompress(frame).view(np.uint16).reshape(1024, 1024)
```

Pass a positive number as `error` for a fixed absolute bound, or a percentage
string for a relative bound. `None` and zero are lossless. Full `LINEAR`, `LOG`
and `SQRT` recipes are also accepted.

```python
absolute = geozl.graph(tile, "planar>zigzag>entropy", error=2)
relative = geozl.graph(tile, "planar>zigzag>entropy", error="1%")
```

`profile` returns a list-like result. Print it to see the input geometry,
benchmark settings and ranked graphs together.

```python
rows = geozl.profile(tile, prior=None)
print(rows)
best = rows[0]["graph"]
```

The documented codecs, lossless and lossy, are reachable two ways. `geozl.graph`
takes a recipe string and builds the graph for you, and `geozl.compress` runs one
tile through it. `geozl.lossless` and `geozl.lossy` hand you the individual nodes,
apart from `lanes` and `planar_zigzag_lanes`, to place in an OpenZL graph yourself, and `geozl.register_decoders` teaches a
`DCtx` to read what they wrote.

Prebuilt wheels are published for Linux x86-64 with CUDA support and macOS arm64
with Metal support. Device decode needs a compatible GPU. A source install on
another platform has to build the native library first; see the build
instructions in the repository.

The full documentation, the codec catalogue and the wire format live in the
[repository](https://github.com/asterisk-labs/geozl).
