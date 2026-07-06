# Adding a codec

A codec is a self contained folder under `core/src/`, shaped like a standard codec
in OpenZL's own `codecs/`, so it copies out whole if it ever graduates upstream.
The fastest start is to copy the nearest existing codec, `planar/` for a one to
one shape, `deinterleave/` for anything else, and adapt it.

It has two parts. A kernel, the transform in pure C with no OpenZL types. A
binding, the OpenZL typed encoder and decoder that wrap the kernel and own the
codec header.

## The CTid

The CTid is the codec's only identity on the wire, names never travel. A reader
decodes a frame only if it has a decoder registered for every CTid the frame
uses. A published CTid is frozen, its decode never changes, and new behaviour
takes a new CTid rather than altering an old one. A reader that meets a newer
CTid fails with a missing decoder, it never decodes wrong.

Every id lives in `core/include/geozl/ctids.h`, lossless in the `0x72D70x` band,
near lossless in `0x72D78x`. Take the next free one, never reuse or renumber. The
format contract is in `SPEC.md`.

## The codec header

The codec header is the only channel from encoder to decoder, the decoder never
sees a local param. Whatever the decode needs per tile is written on encode with
`ZL_Encoder_sendCodecHeader` and read on decode with `ZL_Decoder_getCodecHeader`,
inline in the two bindings. Keep it minimal and fixed. A predictor carries a four
byte width, quant_linear its dtype and scale, deinterleave nothing.

If the codec is exposed to Python, the Python side writes the same layout with
`struct`, since it cannot call the C binding. Both sides follow the codec's
`spec.md`, and the cross reader test catches a disagreement.

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
`common/`. Any other shape carries its own `graph_foo.h`, like `deinterleave/`.

## Wiring

On the C side.

1. `ctids.h`, add the CTid in the right band.
2. `decoder_registry.c`, include the decode binding and register its decoder.
3. `encoder_registry.c`, add the node builder and declare it in `geozl.h`. A width
   predictor is one line, another codec writes a small builder that registers the
   encoder and attaches its params.
4. `core/CMakeLists.txt`, add the codec name to `GEOZL_CODECS`. The foreach picks
   up its four files by the naming convention. An odd extra source, like the
   wp_static trainer, is appended after the loop.

On the Python side, if the codec is exposed there.

5. `_ffi.py`, add the kernel signatures to the cdef.
6. A module under `lossless/` or `lossy/`, and its decoder in that package's
   `register_decoders`.

Then list the codec in the README, its call, CTid, and one line on what it does.
That table is where a user finds it.

## Verifying

Syntax check the binding against the real OpenZL headers.

```
gcc -fsyntax-only -std=c11 -Icore/include -Icore/src -Iextern/openzl/include \
    core/src/foo/decode_foo_binding.c
```

Round trip the kernel in pure C across every element width and the edge tile
shapes, bit exact for lossless, within the bound for near lossless.

Cross reader is the real proof. A frame encoded in C decodes in Python and the
reverse, since both implement the same CTid, header, and kernel.

## Checklist

- [ ] folder with the four files, plus `graph_foo.h` if the shape is not one to one
- [ ] CTid in `ctids.h`, in the right band
- [ ] decoder registered in `decoder_registry.c`
- [ ] node builder in `encoder_registry.c` and `geozl.h`
- [ ] sources building in `core/CMakeLists.txt`
- [ ] cdef and module on the Python side, if exposed
- [ ] listed in the README codec table
- [ ] round trip and cross reader pass