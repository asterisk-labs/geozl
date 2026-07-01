# Adding a codec

A codec is three parts. A **kernel**, the transform in pure C with no OpenZL
types. A **binding**, the OpenZL typed encoder and decoder that wrap the kernel.
A **graph header**, the node's stream types and its transform id. Each codec is
a self contained folder under `core/src/` that shares no code with the others,
so it copies out whole.

The transform id (CTid) is the only identity a frame carries. A reader decodes a
frame only if it has registered a decoder for every CTid the frame references.
Names never travel. One rule follows from that and outranks every convenience
below.

## The CTid contract

A published CTid is frozen. Once a frame written with it exists its decode can
never change, and a reader keeps that decoder even after it stops encoding with
it. New behaviour takes a new CTid, never a mutation of an old one. An old
reader cannot read a CTid minted after it, and that is the only direction that
breaks. It breaks loudly, a missing decoder error, never silent corruption.

geozl's ids sit in the `0x72D700`-`0x72D7FF` block, high enough to stay clear of
OpenZL's standard node ids. Predictors take the `0x72D70x` band, quantizers take
`0x72D78x`. Take the next free id in the right band, never reuse or renumber.

| CTid       | codec        | family    |
|------------|--------------|-----------|
| `0x72D701` | delta_w      | lossless  |
| `0x72D702` | delta_n      | lossless  |
| `0x72D703` | planar       | lossless  |
| `0x72D780` | quant_linear | lossy     |

## Files you create

For a codec `foo`, make `core/src/foo/` after an existing folder. delta_w is the
model for a predictor, quant_linear for a quantizer.

**`graph_foo.h`** is the node definition, included by both bindings.

```c
#define FOO_CTID 0x72D70?            // next free id, see the table
#define FOO_PARAM_WIDTH 1           // local int param key, predictors only

#define FOO_GRAPH                                            \
    {                                                        \
        .CTid           = FOO_CTID,                          \
        .inStreamType   = ZL_Type_numeric,                   \
        .outStreamTypes = ZL_STREAMTYPELIST(ZL_Type_numeric) \
    }
```

A quantizer that carries a typed payload declares its wire enum here too, the
way quant_linear declares `ql_dtype`. The enum order is the wire value and
freezes with the CTid.

**`encode_foo_kernel.{h,c}`** and **`decode_foo_kernel.{h,c}`** are the transform
and its inverse, pure C, no OpenZL. The internal names are `foo_encode` and
`foo_decode`. This is where the SIMD and the dispatch by element width live. The
encode kernel is reached from Python over cffi, the decode kernel from both the
C decoder and Python.

**`decode_foo_binding.{h,c}`** is the OpenZL typed decoder. The header declares
the function and a `static const ZL_TypedDecoderDesc foo_decoder_desc` holding
`.gd = FOO_GRAPH`, `.transform_f`, and a `.name` of `geozl.<family>.foo`. The
body reads any per tile parameter with `ZL_Decoder_getCodecHeader`, returns
`ZL_ErrorCode_corruption` if the header size or the stream shape is wrong,
allocates the output with `ZL_Decoder_create1OutStream`, calls the decode
kernel, and commits.

**`encode_foo_binding.{h,c}`** is the OpenZL typed encoder, declaring
`foo_encoder_desc`. It exists for symmetry and for a later C encode path or
upstream graduation. It is not compiled today, since geozl encodes from Python
through openzl.ext.

**`spec.md`** is the data contract in one page. What the codec predicts or
quantizes, the codec header layout, the element widths it accepts, and for a
quantizer the error bound and its mode.

## Files you edit

Six wiring points, in order.

1. **`core/include/geozl/kernels.h`**, declare the two forwarders.

   ```c
   GEOZL_API void geozl_foo_encode(void* dst, const void* src,
                                   size_t width, size_t nb_elts, size_t elt_width);
   GEOZL_API void geozl_foo_decode(void* dst, const void* src,
                                   size_t width, size_t nb_elts, size_t elt_width);
   ```

   A quantizer uses its own signature. quant_linear takes `double scale, int
   dtype, size_t nb_elts` in place of the width.

2. **`core/src/kernels.c`**, add the forwarder bodies, one line each, calling the
   internal kernel name.

3. **`core/src/register_lossless.c`** or **`core/src/register_lossy.c`**, include
   `foo/decode_foo_binding.h` and add
   `ZL_DCtx_registerTypedDecoder(dctx, &foo_decoder_desc)` with the same error
   check as its neighbours.

4. **`core/CMakeLists.txt`**, add the sources. `GEOZL_DECODE_SOURCES` gets
   `src/foo/decode_foo_binding.c` and `src/foo/decode_foo_kernel.c`.
   `GEOZL_KERNEL_SOURCES` gets `src/foo/encode_foo_kernel.c` and
   `src/foo/decode_foo_kernel.c`. The encode binding goes in neither.

5. **`bindings/python/geozl/_ffi.py`**, add the two declarations to the cffi
   cdef block, matching the kernels.h signatures exactly.

6. **`bindings/python/geozl/lossless.py`** or **`lossy.py`**, wire the Python
   side. A predictor is one `_make_codec` call and one `_PredictorNode`
   subclass, then its decoder into `register_decoders` and its public names into
   `__all__`. A quantizer follows the quant_linear shape, a bespoke encoder that
   carries its parameters, a decoder, a head of graph builder, the wire dtype
   map if it has one, and the same two updates.

Then expose it. Add the codec to its family table in the [README](../README.md),
its call, its CTid, and a one line summary, and add the id to the allocation
table above so the registry stays complete.

## Lossless versus lossy

| aspect            | lossless                              | lossy                              |
|-------------------|---------------------------------------|------------------------------------|
| reversibility     | bit exact, `decode(encode(x)) == x`   | bounded, within the declared error |
| CTid band         | `0x72D70x`                            | `0x72D78x`                         |
| position in graph | anywhere before entropy               | head of graph, exactly one         |
| codec header      | width, 4 bytes LE                     | codec specific, dtype plus scale   |
| Python factory    | `_make_codec` in lossless.py          | bespoke, see lossy.py              |
| content checksum  | left on                               | off, the hash assumes bit exact    |

## Invariants

The codec header is the only channel from encoder to decoder. The decoder never
sees a local param, so whatever the decode needs per tile, the width for a
predictor, the dtype and scale for quant_linear, is written there on encode and
read back with `ZL_Decoder_getCodecHeader`. Keep it minimal and fixed layout.

A predictor's inverse must vectorize. delta_w, delta_n and planar invert with a
running sum or a plain vector add along one axis, which SIMD runs at streaming
rate. A predictor whose inverse branches per pixel, MED or Paeth, decodes
serially at a fraction of that and does not belong here, whatever it wins on
ratio.

A lossy frame carries exactly one quantizer, at the head, with content checksum
off, since the checksum assumes a bit exact round trip. The error bound rides in
the codec header and the decoder needs nothing else.

## Verifying

Syntax check the binding against the real OpenZL headers first.

```
gcc -fsyntax-only -std=c11 -Icore/include -Iextern/openzl/include \
    core/src/foo/decode_foo_binding.c
```

Round trip in Python. Build a graph with the new node over openzl.ext, encode a
tile, register the decoder into a DCtx and decode. Assert equality for a
lossless codec or the bound for a lossy one. Cover every element width the
kernel claims and the edge tile dimensions.

Cross reader. A frame encoded from Python must decode through the C library, and
a frame encoded in C must decode in Python, since both register the same CTid.
That round trip across the two readers is the real proof that the CTid, the
codec header layout, and the kernels agree.