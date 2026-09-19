# Low-level graphs with `openzl.ext`

The recipe grid in `geozl.graph` covers one predictor, one terminal and optional
nodata and quantizer stages. For anything else (another OpenZL backend, a different
stage order, two lanes of a complex raster, a custom mask graph) place geozl nodes in
an `openzl.ext` graph yourself. Sources: `bindings/python/geozl/lossless/`,
`lossy/`, `_codec.py`, and the tests `test_nodes.py`, `test_codecs.py`,
`test_cross_reader.py`, `test_deinterleave.py`.

## Contents

1. How composition works
2. geozl nodes
3. Compression context settings you must set
4. Examples (all verified on 0.16.0)
5. Decoding
6. Raw payload helpers
7. Codecs without a Python node

## 1. How composition works

Every geozl node is a factory. Calling it with a compressor and its successor
registers the encoder and returns an `openzl.ext.GraphID`:

```text
graph_id = Node(args)(compressor, successor)
```

Build from the tail toward the head. A successor can be a bound `GraphID`
(`zl.graphs.Store()(c)`) or an unbound graph (`zl.graphs.Compress()`); geozl
parameterizes unbound ones for you. OpenZL built-ins follow the same pattern:
`zl.nodes.Zigzag()(c, successor)`, `zl.graphs.Entropy()(c)`, `zl.graphs.Zstd()(c)`,
`zl.graphs.FieldLz()(c)`, `zl.graphs.Store()(c)`, `zl.graphs.Compress()(c)`.

## 2. geozl nodes

| Node | In | Out | Notes |
| --- | --- | --- | --- |
| `geozl.lossless.DeltaW(width)` | numeric | numeric | no planes |
| `geozl.lossless.DeltaN(width, planes=1)` | numeric | numeric | signed residuals, follow with `zl.nodes.Zigzag()` before a packer |
| `geozl.lossless.Planar(width, planes=1)` | numeric | numeric | signed residuals |
| `geozl.lossless.PlanarZigzag(width, planes=1)` | numeric | numeric | already zigzagged |
| `geozl.lossless.PlanarZigzagPfor(width, planes=1)` | numeric | **serial** | successor must accept serial, e.g. `zl.graphs.Store()` |
| `geozl.lossless.Med(width, planes=1)` | numeric | numeric | |
| `geozl.lossless.Average(width, planes=1)` | numeric | numeric | |
| `geozl.lossless.WpStatic(width, planes=1)` | numeric | numeric | trains weights per stream |
| `geozl.lossless.Deinterleave()` | numeric | numeric x 2 | even and odd positions; both lanes go to the one successor |
| `geozl.lossless.Nodata(width, value=None, dtype=None)` | numeric | values numeric, mask numeric (1 byte) | called as `(c, values_successor, mask_successor)` |
| `geozl.lossless.Pfor()` | numeric | **serial** | successor must accept serial |
| `geozl.lossy.QuantLinear(recipe, dtype)` | numeric | numeric | full recipe string, e.g. `"LINEAR:MAX_ERROR=0.5"` |
| `geozl.lossy.QuantLog(recipe, dtype)` | numeric | numeric | `"LOG:MAX_ERROR=1%"` |
| `geozl.lossy.QuantSqrt(recipe, dtype)` | numeric | numeric | recipe **must** carry `A=` and `B=` (use `fit_noise(...).recipe(k)`) |

Low-level quantizers resolve their grid from each stream they encode. There is no
frozen domain and no cross-tile check as in `geozl.graph`; float LINEAR `INDEX` grids
can differ between tiles while every tile keeps its bound.

Each node class has a matching decoder (`PlanarDecoder`, `QuantLinearDecoder`, ...),
all registered by `geozl.register_decoders`.

## 3. Compression context settings you must set

```python
cc = zl.CCtx()
cc.ref_compressor(c)
cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)   # without it: OpenZL error code 61
cc.set_parameter(zl.CParam.ContentChecksum, 2)                      # only with a quantizer; 2 disables
```

A quantizer under the default content checksum raises inside the encoder
(`disable ContentChecksum on the CCtx`), surfacing as `RuntimeError: OpenZL error code: 1`.
Decoding without `geozl.register_decoders(dctx)` fails with error code 30.

Input goes in as one numeric stream: `zl.Input(zl.Type.Numeric, np.ascontiguousarray(arr).reshape(-1))`.

## 4. Examples (all verified on 0.16.0)

### Helpers used below

```python
import numpy as np
import openzl.ext as zl
import geozl

def encode(build, arr, lossy=False):
    c = zl.Compressor()
    c.select_starting_graph(build(c))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    if lossy:
        cc.set_parameter(zl.CParam.ContentChecksum, 2)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, np.ascontiguousarray(arr).reshape(-1))]))

def decode(frame, dtype):
    d = zl.DCtx()
    geozl.register_decoders(d)
    return d.decompress(frame)[0].content.as_nparray().view(dtype)
```

### Lossless: planar, zigzag, entropy

`zl.graphs.Entropy` takes 1 or 2 byte elements; for 4 and 8 byte rasters end in
`zl.graphs.FieldLz()`, `zl.graphs.Zstd()` or `geozl.lossless.Pfor()` instead.

```python
W = tile16.shape[1]                   # a uint16 raster
frame = encode(lambda c: geozl.lossless.Planar(W)(c, zl.nodes.Zigzag()(c, zl.graphs.Entropy()(c))), tile16)
assert np.array_equal(decode(frame, tile16.dtype).reshape(tile16.shape), tile16)
```

### Fused planar, zigzag and PFOR over a cube

```python
B, Y, X = cube.shape
frame = encode(lambda c: geozl.lossless.PlanarZigzagPfor(X, planes=B)(c, zl.graphs.Store()(c)), cube)
```

### NoData, quantizer, predictor, packer

```python
W = tile.shape[1]                     # float32 tile with -9999 holes

def build(c):
    tail = geozl.lossless.Pfor()(c, zl.graphs.Store())
    pred = geozl.lossless.PlanarZigzag(W)(c, tail)
    quant = geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.float32)(c, pred)
    nd = geozl.lossless.Nodata(W, value=-9999.0, dtype=np.float32)
    return nd(c, quant, zl.graphs.Compress())

frame = encode(build, tile, lossy=True)
back = decode(frame, np.float32).reshape(tile.shape)
```

### Complex SAR: split real and imaginary before predicting

```python
def encode_complex(z):
    w = z.shape[1]
    comp = geozl.lossless.component_dtype(z.dtype)          # complex64 -> float32
    flat = np.ascontiguousarray(z).reshape(-1).view(comp)   # [re, im, re, im, ...]

    def build(c):
        g = zl.graphs.Compress()(c)
        g = zl.nodes.Zigzag()(c, g)
        g = geozl.lossless.DeltaW(w)(c, g)
        return geozl.lossless.Deinterleave()(c, g)          # both lanes share the successor

    return encode(build, flat)

z_back = decode(encode_complex(z), z.dtype)                 # the lanes rejoin as the complex layout
```

`DeltaW(w)` over a lane of `w` samples per row is correct because each lane holds one
component per pixel.

## 5. Decoding

- Python decoders: `zl.DCtx()` plus `geozl.register_decoders(dctx)`, then
  `dctx.decompress(frame)[0].content.as_nparray()`. The array is typed by element width
  as unsigned; apply `.view(dtype)`.
- C decoders: `geozl.decompress(frame)` reads low-level frames too (flat `uint8`).
- Both sides implement the same wire format and are cross-tested: frames written by C
  decode in Python and the reverse, byte for byte on lossy frames.

## 6. Raw payload helpers

No OpenZL frame, only the codec payload, useful for experiments and tests:

```python
from geozl.lossless import pfor, planar_zigzag_pfor

packed = pfor.encode(values)                          # 1-D numeric array
same = pfor.decode(packed, values.size, values.dtype)
pfor.bound(n, itemsize)                               # worst-case size, 0 for invalid geometry

payload = planar_zigzag_pfor.encode(raster, width, planes=1)
back = planar_zigzag_pfor.decode(payload, raster.size, raster.dtype, width, planes=1)
```

`planar_zigzag_pfor.encode` produces the same bytes as `planar_zigzag` followed by `pfor`.

## 7. Codecs without a Python node

| Codec | CTid | Status |
| --- | --- | --- |
| `binoffset`, `intmult`, `floatquant`, `floatmult` | `0x72D708` to `0x72D70B` | ported from pcodec in 0.7.0; Python bindings removed in 0.8.0; C nodes and decoders stay so old frames decode |
| `blocked_transpose_zstd` | `0x72D70E` | work in progress: C node `geozl_node_blocked_transpose_zstd(c, blockSize)` and a recipe terminal, no Python node or docs |

`geozl.decompress` and `geozl_register_decoders` (C) decode all of them.
`geozl.register_decoders` (Python) does not.
