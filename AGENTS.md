# GeoZL repository guidance

GeoZL is a C11 codec library with Python bindings, built on the pinned OpenZL
submodule in `extern/openzl`.

## Working agreements

- Treat `extern/openzl` as an upstream submodule. Do not edit files inside it as
  part of an ordinary GeoZL change; update the pinned revision only when the task
  explicitly requires an OpenZL upgrade.
- Treat files under `core/build*`, Python caches, and staged native libraries
  under `bindings/python/geozl/_lib` as build artifacts, not source. The HTML,
  CSS, JavaScript, and media under `docs/` are hand-maintained source files.
- Preserve frame compatibility. Never renumber or reuse a shipped CTid. A wire
  format change needs a new CTid or an explicitly compatible decoder path, plus
  an updated codec spec, compatibility documentation, and golden-frame coverage.
- Decoders process untrusted frames. Validate stream types, lengths, counts,
  arithmetic, and allocation sizes before reading or writing buffers.
- Keep the C kernels independent of Python and, where the current split uses one,
  independent of OpenZL. Put OpenZL orchestration in bindings and registries.
- Keep the stable C source API backward compatible; before 1.0, ABI changes still
  require callers to rebuild.

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
