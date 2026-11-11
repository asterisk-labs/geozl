# GeoZL repository guidance

GeoZL is a C11 codec library with Python bindings, built on the pinned OpenZL
submodule in `extern/openzl`.

## Working agreements

- Treat `extern/openzl` as an upstream submodule. Do not edit files inside it as
  part of an ordinary GeoZL change; update the pinned revision only when the task
  explicitly requires an OpenZL upgrade.
- Treat files under `core/build*`, Python caches, and staged native libraries
  under `bindings/python/geozl/_lib` as build artifacts, not source. The HTML,
  CSS, JavaScript, and media under `docs/` are source files.
- GeoZL is pre-1.0 and reads only the current wire format. Do not keep legacy
  decoder paths, aliases or compatibility shims for older frames.
- Never renumber a CTid or reuse it for another codec, and record a retired one
  in `core/include/geozl/ctids.h`. Before 1.0, an intentional in-place layout
  replacement may keep its CTid when `docs/compatibility.md` records the break.
  Update the codec spec and regenerate the current golden corpus with
  `tools/write_golden.py`; do not keep the old decoder.
- Decoders process untrusted frames. Validate stream types, lengths, counts,
  arithmetic, and allocation sizes before reading or writing buffers.
- Keep the C kernels independent of Python and, where the current split uses one,
  independent of OpenZL. Put OpenZL orchestration in bindings and registries.
- The C source API in `geozl/geozl.h` and `geozl/gpu.h` may change before 1.0.
  Remove a superseded declaration instead of keeping an alias, and expect
  callers to rebuild after an update.
- On CUDA, `geozl_gpu_decode` returns once its work is queued and a later call on
  the same stream may leave it running. Call `geozl_gpu_synchronize` before
  freeing or reusing a target, and keep a stream alive until its work has finished.

## Validation

- Run the narrowest relevant tests while iterating, then `make test` for changes
  that affect behavior across the C and Python layers.
- Run `ruff check .` and `mypy` after Python changes.
- Run the sanitizer target for decoder, buffer, wire-format, or other memory-safety
  changes. Use the fuzz and exhaustive targets when the change warrants them; they
  are intentionally not part of the normal fast loop.

Detailed GeoZL usage, codec-development, compatibility, and debugging procedures
live in the repository's `geozl` skill. Load the relevant reference from that skill
instead of expanding this always-on file with task-specific instructions.
