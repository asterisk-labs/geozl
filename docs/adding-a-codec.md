# Adding a codec

A geozl codec is three things: a kernel (the transform, pure C), a binding (the
OpenZL typed encoder and decoder around the kernel), and a graph header (the
node type and its transform id). Each codec lives in its own folder under
`core/` and shares no code with the others, so a folder is copy-pasteable on its
own.

The transform id (CTid) is the only identity that travels in a frame. A reader
decodes a frame only if it has registered a decoder for every CTid the frame
references. Names never travel. This has one consequence that overrides
convenience.

## The CTid rule

A published CTid is frozen. Once a frame written with it exists, its decode
behaviour can never change, and a reader keeps the decoder for it even after it
stops encoding with it. New behaviour takes a new CTid, it never mutates an
existing one. An old reader cannot read a CTid added after it, that is the only
direction that breaks, and it breaks loudly with a missing-decoder error, not
with silent corruption.

CTids live in the experimental range `0x72D700`-`0x72D7FF`. Current allocation:

| CTid       | codec        | family       |
|------------|--------------|--------------|
| 0x72D701   | delta_w      | experimental |
| 0x72D702   | delta_n      | experimental |
| 0x72D703   | planar       | experimental |
| 0x72D780   | quant_linear | lossy        |

Lossless predictors take `0x72D70x`, lossy codecs take `0x72D78x`, by
convention. Take the next free id in the right band. Do not reuse, do not
renumber.

## Files you create

For a codec `foo`, create `core/foo/` mirroring an existing folder. delta_w is
the reference for a lossless predictor, quant_linear for a lossy codec.

### `core/foo/graph_foo.h`

The shared node definition, included by both bindings.

```c
#define FOO_CTID 0x72D70?            // next free id, see the table
#define FOO_PARAM_WIDTH 1            // local int param key, predictors only

#define FOO_GRAPH                                        \
    {                                                    \
        .CTid           = FOO_CTID,                      \
        .inStreamType   = ZL_Type_numeric,              \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }
```

A lossy codec that carries a typed payload puts its wire enum here too, the way
quant_linear declares `ql_dtype`. The enum order is the wire value and is frozen
with the CTid.

### `core/foo/encode_foo_kernel.{h,c}` and `core/foo/decode_foo_kernel.{h,c}`

The transform and its inverse, pure C, no OpenZL types. Internal names
`foo_encode` and `foo_decode`. This is where the SIMD lives and where the
dispatch by element width happens. The encode kernel is reached by Python over
cffi, the decode kernel by both the C decoder and Python.

### `core/foo/decode_foo_binding.{h,c}`

The OpenZL typed decoder. The header declares the decode function and a
`static const ZL_TypedDecoderDesc foo_decoder_desc` with `.gd = FOO_GRAPH`,
`.transform_f`, and `.name`. The decoder recovers any per-tile parameter from
the codec header, validates `eltWidth` and `numElts` and returns
`ZL_ErrorCode_corruption` if they do not hold, creates its output, calls the
decode kernel, commits.

### `core/foo/encode_foo_binding.{h,c}`

The OpenZL typed encoder, declaring `foo_encoder_desc`. Kept for symmetry and
for a future C encode path or upstream graduation. It is not compiled today,
geozl encodes from Python through openzl.ext. See the CMake note below.

### `core/foo/spec.md`

The codec's data contract in one page: what it predicts or quantizes, the codec
header layout, the supported element widths, and for a lossy codec the error
bound and its mode.

## Files you edit

Six wiring points, in this order.

1. **`core/include/geozl/kernels.h`** declare the two forwarders.

   ```c
   GEOZL_API void geozl_foo_encode(void* dst, const void* src,
                                   size_t width, size_t nb_elts, size_t elt_width);
   GEOZL_API void geozl_foo_decode(void* dst, const void* src,
                                   size_t width, size_t nb_elts, size_t elt_width);
   ```

   A lossy codec uses its own signature, quant_linear takes `double scale, int
   dtype, size_t nb_elts` instead of width.

2. **`core/kernels.c`** add the forwarder bodies, one line each, calling the
   internal kernel name.

3. **`core/register_experimental.c`** or **`core/register_lossy.c`** include
   `foo/decode_foo_binding.h` and add
   `ZL_DCtx_registerTypedDecoder(dctx, &foo_decoder_desc)` with the same error
   check as the others.

4. **`core/CMakeLists.txt`** add the new sources to both lists.
   `GEOZL_DECODE_SOURCES` gets `foo/decode_foo_binding.c` and
   `foo/decode_foo_kernel.c`. `GEOZL_KERNEL_SOURCES` gets
   `foo/encode_foo_kernel.c` and `foo/decode_foo_kernel.c`. The encode binding
   goes in neither.

5. **`python/geozl/_ffi.py`** add the two forwarder declarations to the cffi
   cdef block, matching the kernels.h signatures exactly.

6. **`python/geozl/experimental.py`** or **`python/geozl/lossy.py`** wire the
   Python side. For an experimental predictor this is one `_make_codec` call and
   one `_PredictorNode` subclass, then the decoder goes into `register_decoders`
   and the public names into `__all__`. For a lossy codec, follow the
   quant_linear shape: a bespoke encoder that carries its parameters, a decoder,
   a head-of-graph builder function, the wire dtype map if it has one, and the
   same `register_decoders` and `__all__` updates.

## Experimental versus lossy

| aspect            | experimental                          | lossy                                  |
|-------------------|---------------------------------------|----------------------------------------|
| reversibility     | bitexact, decode(encode(x)) == x      | bounded, within the declared error     |
| CTid band         | 0x72D70x                              | 0x72D78x                               |
| position in graph | anywhere before entropy               | head of graph, exactly one             |
| codec header      | width, 4 bytes LE                     | codec specific, e.g. dtype + scale     |
| Python factory    | `_make_codec` in experimental.py      | bespoke, see lossy.py                  |
| content checksum  | left on                               | off, the hash assumes bitexact         |

## Invariants

The codec header is the only channel from encoder to decoder. The decoder never
sees local params, so anything per-tile that the decode needs (width for a
predictor, dtype and scale for quant_linear) is written there on encode and read
back with `ZL_Decoder_getCodecHeader`. Keep it minimal and fixed-layout.

A predictor must stay scannable. Only an inverse that is associative along the
left-neighbour chain vectorizes, which is why delta_w, delta_n and planar are
the predictors that exist. A serial inverse such as MED or Paeth would decode at
a fraction of the rate and must not be added as a predictor, whatever it buys on
ratio.

A lossy frame carries exactly one lossy codec, at the head, and turns content
checksum off, since the checksum assumes a bitexact round trip. The error bound
rides in the codec header, the decoder needs no side information.

## Verifying

Syntax check the binding against the real OpenZL headers before anything else:

```
gcc -fsyntax-only -std=c11 -Icore/include -Iextern/openzl/include \
    core/foo/decode_foo_binding.c
```

Round trip in Python: build a graph with the new node over openzl.ext, encode a
tile, register the decoder into a DCtx and decode, assert equality for a
lossless codec or the bound for a lossy one. Exercise every element width the
kernel claims to support and the edge tile dimensions.

Cross-reader: a frame written from Python must decode in the rumi C reader and a
frame written by rumi must decode in Python, since both register the same CTid.
That round trip across the two readers is the real test that the CTid, the codec
header layout and the kernels agree.
