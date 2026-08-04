# geozl

Python bindings for [geozl](https://github.com/asterisk-labs/geozl), geospatial
codec extensions for [OpenZL](https://github.com/facebook/openzl).

```python
import numpy as np
import geozl

tile = np.fromfile("b04.raw", dtype=np.uint16).reshape(1024, 1024)

frame = geozl.compress(tile, method="planar>zigzag>transpose>entropy")
back = geozl.decompress(frame, dtype="uint16", width=1024)
```

Twelve codecs, lossless and lossy, reachable two ways. `geozl.compress` takes a
recipe string and builds the graph for you. `geozl.lossless` and `geozl.lossy`
hand you the individual nodes to place in an OpenZL graph yourself, and
`geozl.register_decoders` teaches a `DCtx` to read what they wrote.

Wheels carry a prebuilt native library. A source install has to build it first,
see the build instructions in the repository.

The full documentation, the codec catalogue and the wire format live in the
[repository](https://github.com/asterisk-labs/geozl).
