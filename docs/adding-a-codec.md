# Adding a codec

A codec is a self contained folder under `core/src/`, shaped like a standard codec
in OpenZL's own `codecs/`, so it copies out whole if it ever graduates upstream.
The names are load bearing, `core/CMakeLists.txt` globs for
`src/<name>/encode_<name>_binding.c` and takes the codec from there.

## The CTid

The CTid is the codec's only identity on the wire, names never travel. A reader
decodes a frame only if it has a decoder registered for every CTid the frame
uses. A published CTid is frozen, its decode never changes, and new behaviour
takes a new CTid rather than altering an old one. A reader that meets a newer
CTid fails with a missing decoder, it never decodes wrong.

Every id lives in `core/include/geozl/ctids.h`, lossless in `0x72D700` to
`0x72D77F`, near lossless in `0x72D780` to `0x72D7FF`. The band is not a
convention, `geozl_ctid_is_lossy` reads the kind straight off the id, so an id in
the wrong band makes the codec lie about whether it destroys data. **Take the next
free one, never reuse or renumber.**

## The codec header

The codec header is the only channel from encoder to decoder, the decoder never
sees a local param. Whatever the decode needs per tile is written on encode with
`ZL_Encoder_sendCodecHeader` and read on decode with `ZL_Decoder_getCodecHeader`,
inline in the two bindings. Keep it minimal! For instace, most predictors carry
a four byte width, wp_static that width plus a shift and four coefficients,
quant_linear its dtype and scale, deinterleave nothing.

## The folder

- `encode_foo_kernel.{c,h}`, `decode_foo_kernel.{c,h}`. The transform and its
  inverse, pure C, no OpenZL, dispatch by element width. Internal names
  `foo_encode` and `foo_decode`. They depend only on `<stdint.h>`, `<stddef.h>`,
  and `<string.h>`.
- `encode_foo_binding.{c,h}`, `decode_foo_binding.{c,h}`. The typed encoder and
  decoder. The header declares the function and, as `extern`, the descriptor,
  which is defined in the `.c` so it has one definition and no unused warning. The
  encoder reads its params, creates the output, runs the kernel, writes the
  header, and commits. The decoder reads the header, rejects a wrong size or
  stream shape with `ZL_ErrorCode_corruption`, then runs the kernel and commits.
- `spec.md`. The decoder contract, inputs, header layout, the widths accepted,
  and for a near lossless codec the error bound.

A one numeric in, one numeric out codec reuses `GEOZL_NUM1TO1_GRAPH` from
`common/`, a one in two out reuses `GEOZL_NUM1TO2_GRAPH` from the same place.
Any other shape carries its own `graph_foo.h`, like `quant_linear/`.

## Wiring

On the C side.

1. `ctids.h`, add the CTid in the right band.
2. `decoder_registry.c`, include the decode binding and register its decoder.
3. `encoder_registry.c`, add the node builder and declare it in `geozl.h`. A width
   predictor is one line, another codec writes a small builder that registers the
   encoder and attaches its params.
4. `core/CMakeLists.txt`, nothing. A `CONFIGURE_DEPENDS` glob takes any
   `src/<name>/` holding an `encode_<name>_binding.c` for a codec and re-globs on
   build, so a new folder is picked up on its own and the four files come with it
   by the naming convention. A source outside that convention, like the wp_static
   trainer, is the one case that needs a line, appended after the loop.

## Verifying

Syntax check the binding against the real OpenZL headers. They come from the
submodule, so a fresh clone runs `make submodules` first.

```
gcc -fsyntax-only -std=c11 -Icore/include -Icore/src -Iextern/openzl/include \
    core/src/foo/decode_foo_binding.c
```

Round trip the kernel in pure C across every element width and the edge tile
shapes, bit exact for lossless, within the bound for near lossless.

Cross reader is the real proof. A frame encoded in C decodes in Python and the
reverse, since both implement the same CTid, header, and kernel.

`make test` runs the suite, `make test-san` the same under ASan and UBSan, and
`make fuzz` builds the decode fuzzer and its corpus.

## Checklist

- [ ] folder with the four files, plus `graph_foo.h` if the shape is not one to one
      or one to two
- [ ] CTid in `ctids.h`, in the right band
- [ ] decoder registered in `decoder_registry.c`
- [ ] node builder in `encoder_registry.c` and `geozl.h`
- [ ] `core/CMakeLists.txt` only if the codec has a source outside the convention
- [ ] cdef and module on the Python side, if exposed
- [ ] listed in the README codec table
- [ ] round trip and cross reader pass