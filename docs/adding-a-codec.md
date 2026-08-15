# Adding a codec

A codec is a self contained folder under `core/src/`, shaped like a standard codec
in OpenZL's own `codecs/`, so it copies out whole if it ever graduates upstream.
The names are load bearing, `core/CMakeLists.txt` globs for
`src/<name>/encode_<name>_binding.c` and takes the codec from there.

## The CTid

The CTid is the codec's only identity on the wire; names never travel. Beginning
with 0.13.0, later releases continue to decode every released GeoZL frame.
Existing CTids are never reassigned, and an incompatible layout takes a new
one. Encoder output may change, and older readers need not support newer codecs
or compatible extensions. This guarantee covers GeoZL codec data; OpenZL
container compatibility follows OpenZL.

Every id lives in `core/include/geozl/ctids.h`, lossless in `0x72D700` to
`0x72D77F`, near lossless in `0x72D780` to `0x72D7FF`. The band is not a
convention, `geozl_ctid_is_lossy` reads the kind straight off the id, so an id in
the wrong band makes the codec lie about whether it destroys data. Take the next
free one, never reuse or renumber.

## The codec header

The codec header is the only channel from encoder to decoder, the decoder never
sees a local param. Whatever the decode needs per tile is written on encode with
`ZL_Encoder_sendCodecHeader` and read on decode with `ZL_Decoder_getCodecHeader`,
inline in the two bindings. Keep it minimal. Most predictors carry a four byte
width, wp_static that width plus a shift and four coefficients, a quantizer its
dtype, flags and step, nodata the sentinel pattern alone, deinterleave nothing.

## The folder and the wiring

Four pairs of files, and the CTid.

**`encode_foo_kernel.{c,h}`, `decode_foo_kernel.{c,h}`.** The transform and its
inverse, pure C, no OpenZL, dispatch by element width. A codec with one numeric
stream in and one out names them `foo_encode` and `foo_decode`; a codec that
splits a stream names what it does instead, as `deinterleave_split` and
`nodata_mark_nan` do. They return `int`, nonzero for a geometry they refuse,
except where nothing can be refused and `void` is honest, or where a count is
the useful thing to say, which is what the nodata markers return.

Most depend only on `<stdint.h>`, `<stddef.h>` and `<string.h>`. The quantizers
and the wp_static trainer call into libm, which is why `core/CMakeLists.txt`
runs `find_library(GEOZL_LIBM m)`. binoffset and floatmult include `<math.h>`
too, but only for `INFINITY` and `isfinite`, so they cost nothing at link.
Anything that includes `common/scan.h` picks its path with `geozl_simd_has`, so
it needs `common/simd.c` at link time.

**`encode_foo_binding.{c,h}`, `decode_foo_binding.{c,h}`.** The typed encoder and
decoder. Each header declares its function, `EI_geozl_foo` or `DI_geozl_foo`, and
a macro `EI_FOO(id)` or `DI_FOO(id)` that expands to the descriptor. A macro
rather than a variable, so the descriptor is built at the one place it is
registered and there is nothing to keep alive.

The encoder reads its params, creates the outputs, runs the kernel, writes the
header, and commits. The decoder reads the header, rejects a wrong size or
stream shape with `ZL_ErrorCode_corruption`, then runs the kernel and commits.

The graph description inside the descriptor is shared. One numeric in, one
numeric out reuses `GEOZL_NUM1TO1_GRAPH` from `common/`, one in two out reuses
`GEOZL_NUM1TO2_GRAPH`. A codec whose stream types are its own carries a
`graph_foo.h`, which is what the three quantizers do. They are one in and one
out like a predictor, but the output is always an integer stream whatever went
in, so the shared macro does not describe them.

**`spec.md`.** The decoder contract. What a reader has to know to invert the
codec without seeing the encoder, so inputs, header layout, the widths accepted,
and for a near lossless codec the error bound. A simple codec settles into
`### Inputs`, `### Codec Header`, `### Decoding` and `### Outputs`, which the
predictors all use. The quantizers are longer and organised around what they
refuse and where the bound ends, since that is what a reader of those needs.
Follow whichever the codec is closer to rather than padding out headings.

Then three edits, and one non-edit.

1. `ctids.h`, add the CTid in the right band.
2. `decoder_registry.c`, include the decode binding and add a
   `REGISTER(GEOZL_CTID_FOO, DI_FOO)` row to `kDecoders`.
3. `encoder_registry.c`, write the node builder and declare it in `geozl.h`. A
   width predictor is one line through `width_node`. Anything else registers the
   encoder and attaches its params, and refuses at the top whatever a caller can
   get wrong, since that is the one boundary a C caller crosses.
4. `core/CMakeLists.txt`, nothing. A `CONFIGURE_DEPENDS` glob takes any
   `src/<name>/` holding an `encode_<name>_binding.c` for a codec and re-globs on
   build, so a new folder is picked up on its own and the four pairs come with it
   by the naming convention. A source outside that convention needs its own line
   after the loop, where the wp_static trainer, the spec parsers, the sqrt noise
   fit and `common/simd.c` already sit.

## The Python side

Only if the codec is exposed, but read this before deciding it is not, because
the cross reader test below is the strongest check the project has and it needs
both sides.

The kernels are shared, not reimplemented. Python declares them in the `_CDEF`
block of `bindings/python/geozl/_ffi.py` and calls the same shared library the C
reader links. A signature in that block that disagrees with the header is not a
build error. cffi will read a return value that was never written and nothing
downstream notices, so a kernel whose signature changes is two edits, never one,
and `test_cdef.py` diffs the block against the headers to catch exactly this.

What Python adds is the encoder, the decoder and the header packing, in a module
under `lossless/` or `lossy/`. A spatial predictor is one call to
`spatial_predictor` in `_codec.py` and the module is four lines, as
`lossless/planar.py` shows. A quantizer takes `quantizer` from the same place.
Anything else writes its own `CustomEncoder` and `CustomDecoder`, which is what
`lossless/nodata.py` does, and then repeats every check the C binding makes.
Export the pair from the package `__init__.py` and add the decoder to
`_DECODERS`, or `geozl.register_decoders` will not know about it.

## Verifying

Syntax check the binding against the real OpenZL headers. They come from the
submodule, so a fresh clone runs `make submodules` first.

```
gcc -fsyntax-only -std=c11 -Icore/include -Icore/src -Iextern/openzl/include \
    core/src/foo/decode_foo_binding.c
```

Round trip the kernel in pure C across every element width and the edge tile
shapes, bit exact for lossless, within the bound for near lossless. Test the
kernel through the mask or the buffer it produces, not through a value it
returns, or the test only proves the function agrees with itself.

Cross reader is the real proof. A frame encoded in C decodes in Python and the
reverse, since both implement the same CTid, header, and kernel. Add the codec
to `bindings/python/test/test_cross_reader.py`.

`make test` runs the C suite then pytest, `make test-san` the same under ASan
and UBSan, and `make fuzz` builds six harnesses, the decode fuzzer plus the ones
that drive the kernels directly.

## Where the checking goes

The decode binding reads the codec header and the decode kernel reads the
stream, so between them they are handed everything a forged frame can carry.
Which of the two refuses it is not free to choose.

Anything the kernel would compute on has to be checked by the kernel. Every
kernel is exported from `libgeozl_kernels`, which is what the Python bindings
load, so a caller can reach it without ever going through a frame or a binding.
A shift the kernel folds in 64 bits, a bin wider than its element, a dtype used
as a table index, an element width the dispatch does not know, all of that
belongs at the top of the kernel, and the binding checking it as well is fine
but does not count. A width the kernel cannot dispatch is still a call that has
to leave its output buffer readable.

The binding keeps what only it can see. Header length, field layout, the
relation between the header and the stream widths OpenZL reports.

The three quantizers do this with a `quant_<name>_check.h` that both ends
include, which is worth copying when a codec has more than a couple of
preconditions.

## Checklist

- [ ] four file pairs, plus `graph_foo.h` if the stream types are the codec's own
- [ ] CTid in `ctids.h`, in the right band
- [ ] `EI_FOO` and `DI_FOO` descriptor macros in the two binding headers
- [ ] decoder row in `kDecoders`, in `decoder_registry.c`
- [ ] node builder in `encoder_registry.c` and declared in `geozl.h`
- [ ] `core/CMakeLists.txt` only if the codec has a source outside the convention
- [ ] `spec.md`, enough for a reader to invert the codec without the encoder
- [ ] if exposed, the `_CDEF` block, the module, and the `__init__.py` exports
      including `_DECODERS`
- [ ] the catalog, all four, or `test_catalog.py` fails. README table, cards in
      `docs/docs.html`, the `CODECS` array in `docs/assets/js/main.js`, and a page
      under `docs/codecs/`
- [ ] round trip and cross reader pass
