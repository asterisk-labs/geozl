# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pfor`, a lossless terminal for fixed-width integer streams.
- Short Python error forms: a number selects `LINEAR`, a percentage string
  selects `LOG`, and zero selects lossless compression.

### Removed

- `store_lo` from the high-level 2D recipe grid.

## [0.13.1] - 2026-08-15

### Fixed

- Wheels include the required third-party licence notices.
- Linux wheels use the x86-64 baseline advertised by their platform tag.

### Added

- `max_output_size` for limiting Python decompression allocations.
- Compatibility fixtures for frames written by 0.13.0.

### Changed

- CI covers Python 3.11 through 3.14.
- Fuzz replay starts from versioned inputs when its cache is empty.
- Python depends on OpenZL 0.2.x.
- The package classifier moves from Alpha to Beta.

## [0.13.0] - 2026-08-15

### Fixed

- Python decoders accept the plane count in stacked predictor frames.

- Reused lossy graphs reject tiles outside their opening domain instead of
  silently saturating them. Declared NoData values are excluded from the check.

### Added

- Frame compatibility policy for releases beginning with 0.13.0.

- Stable C source API policy and a check for the 0.13 public signatures.

- Private vulnerability reporting instructions in `SECURITY.md`.

- Low-level Python predictor nodes accept `planes`, matching their C nodes.

- `fuzz/binding_fuzzer.c` builds valid frames around fuzzed codec headers and
  streams. `make fuzz-replay` runs saved corpora; `make fuzz-check` runs a new
  campaign and fails when libFuzzer writes a finding.

## [0.12.0] - 2026-08-14

### Added

- Stacked planes. The five predictors that read the row above (`planar`,
  `delta_n`, `average`, `med`, `wp_static`) take a plane count and restart at
  each boundary, so a `(B, Y, X)` cube is predicted band by band. Nothing is
  predicted across planes. `delta_w` only reads to its left and is unchanged.

- `geozl.graph(cube, method)` infers the count from the first axis, so a cube
  needs no new argument. `planes=` overrides it, and `profile` takes it too.

### Changed

- The five codec headers grow an optional trailing `uint32`, written only above
  one plane. A four byte header (thirteen for `wp_static`) still means one
  plane, so frames written by earlier versions decode byte for byte.

### Breaking

- `geozl_node_planar`, `geozl_node_delta_n`, `geozl_node_average`,
  `geozl_node_med` and `geozl_node_wp_static` take a `planes` argument, and
  `geozl_2d_compress_c`, `geozl_2d_graph_open_c` and `geozl_2d_bench_c` take one
  after `width`. Pass `1` for the old behaviour.

## [0.11.0] - 2026-08-08

### Changed

- The nodata codec header is the bit pattern and nothing else. One shape on the
  wire needs no code to tell shapes apart, and no count, since the values stream
  already carries it.

### Breaking

- The `all valid` and `all hole` wire codes are gone. A clean tile now writes a
  mask of all valid, which codes to nothing, and an all hole tile fills to two
  constant streams the backends collapse. 0.10.0 frames using either code no
  longer decode.
- The codec refuses an empty tile, at both readers. It used to write one the
  decoder could not read back.
- `nodata_broadcast` is gone, and with it the sample count the header carried.
  That count was the one number a nodata frame declared about itself, so the
  check that caught an overflowing one goes too, and a forged nodata header has
  nothing left to lie about.

## [0.10.0] - 2026-08-07

### Added

- `geozl.graph`, one built graph for many tiles. `compress` registered every
  node and built the DAG on each call, so a raster cut into 400 tiles paid for
  400 compressors. The handle owns the compressor and the CCtx and frees them
  with `ffi.gc`.
- `geozl_2d_graph_open_c`, `geozl_2d_compress_graph_c` and
  `geozl_2d_graph_close_c`, which are `geozl_2d_compress_c` split in two.
- `categorical` terminal. One pass takes the share the dominant symbol holds and
  routes to `constant` at 1, `field_lz` above 0.95, `entropy` below. A cloud
  mask sits near 0.97 and a land cover map near 0.50, so no fixed backend serves
  both. A function graph, so the frame records the arm and there is nothing to
  register on decode.
- `test_graph.py` and `test_categorical.py`.

### Changed

- The grid is 56 recipes, 32 at one byte per element. `categorical` applies at 1
  or 2 bytes, refused at build rather than in the branch, so it does not apply
  or not by what the tile happens to hold.

### Breaking

- `geozl.compress` takes `graph` and nothing else. The recipe, the width, the
  error recipe and the nodata declaration moved to `geozl.graph`.
- `geozl.decompress` returns flat `uint8`, without `dtype` and `width`. The
  caller finishes with `.view(dtype).reshape(shape)`.
- `geozl.profile` takes `prior`, not `method`. The word named a full recipe in
  one call and a predictor family in the other.
- A lossy graph freezes the plan cut against the raster it was built on, since
  the quantizer parameters are part of the graph. A tile reaching past that
  raster quantizes on the wrong grid, and one holding negatives where the raster
  had none decodes floored at zero. Pass the product, not the first tile.
- `geozl_terminal` codes after `zstd` shift by one. Nothing persists them.

## [0.9.0] - 2026-08-06

### Fixed

- `geozl.profile` reported a frame four bytes under the one `geozl.compress`
  writes, because the bench turned the content checksum off and compress does
  not. The ratio it ranks by is now the size you get, to the byte.
- The timed region covered creating the compressor and the DCtx, registering
  every node, graph and decoder, and refitting a curveless SQRT recipe, all on
  every rep. Compression and decompression are now an open, a run and a close,
  and the bench times reps of the run.
- A rep that finished inside a clock tick left the timing at zero, which macOS
  reaches at a microsecond. The bench falls back to the whole run over `reps`,
  and a throughput that is still unresolvable comes back as `inf`.
- `geozl_2d_bench_c` returned from `reps == 0` and from a failed allocation
  without touching `errCtx`. `geozl.profile` turned `reps=0` into an empty table
  and now raises.

### Added

- `verify` on `geozl.decompress` and `geozl_2d_decompress_c`, default on. Off
  skips both checksum verifications, worth 1 to 30 per cent of decode. It cannot
  add a checksum a frame does not carry.
- `verify` on `geozl.profile`, default off, which moves the decode column only.
- `checksum` on `geozl_2d_bench_c`. A bench that drops it stops measuring the
  frame compress writes. `geozl.profile` leaves it on.
- A `bytes` column on every `geozl.profile` row.

### Breaking

- `geozl_2d_decompress_c` takes `int verify` between `outSize` and `errCtx`.
- `geozl_2d_bench_c` takes `int checksum, int verify` between `reps` and
  `compSize`.

## [0.8.1] - 2026-08-04

### Fixed

- A declared nodata sentinel crossed into C as a `double`, so any int64 or
  uint64 value past 2^53 arrived truncated and masked nothing. It crosses as
  bits now.
- `geozl_row_width` folded a width past `nbElts` to one row instead of refusing
  it, so a forged header decoded into a different raster and still returned
  success. The fold moved to the encode side.
- The sdist reached the README through `../../`, which `pip` refused as a path
  traversal, and `testpaths` and the sdist `include` both named `tests` rather
  than `test`.
- ruff was configured in 0.8.0 and never run, and the notebook called `geozl`
  two cells before importing it. The fuzz seed corpus is unchanged.

### Added

- `test_cross_reader.py`. Every codec is written twice and nothing compared the
  two. Flipping the wire format leaves the 999 tests in `test_codecs.py` green
  and fails 68 of these.
- `test_cdef.py`, which diffs the hand written cdef in `_ffi.py` against the
  headers. ABI mode verifies nothing, so drift corrupts memory instead of
  raising.
- `py.typed` with the public entry points annotated, plus ruff, mypy and a
  `sanitize-full` job in CI. The existing sanitizer job is `FULL=OFF`, which
  leaves `2d.c` uncovered.

### Changed

- The sentinel conversion has one implementation, in `_dtype.py`, and a NaN
  sentinel takes the automatic path rather than matching one payload.
- ruff and mypy read one config each from the repository root, and neither walks
  into `extern`.

### Breaking

- `geozl_2d_compress_c` and `geozl_2d_bench_c` take `uint64_t nodataBits` where
  they took `double nodataValue`.
- A 0.8.0 frame written through the low level API with a row width of 0 or past
  the tile is refused. Frames from `geozl.compress` are unaffected.
- `nodata=3.5` on an integer tile raises instead of truncating, and one outside
  the dtype raises `OverflowError`.

## [0.8.0] - 2026-08-04

### Added

- `quant_log` codec, CTid `0x72D782`. Relative error bound, `LOG:MAX_ERROR=P%`.
- `quant_sqrt` codec, CTid `0x72D783`. Bound follows sensor noise `a + b*x`,
  `SQRT:MAX_ERROR=VN`, with `geozl.lossy.fit_noise` to measure the curve once
  over a stack so neighbouring tiles share a grid.
- `nodata` codec, CTid `0x72D70C`. Splits a tile into values and a validity
  mask in the spirit of GDAL, either by detecting NaN or by a declared sentinel.
- High-level Python API, `geozl.profile`, `geozl.compress` and
  `geozl.decompress`. Profile ranks the candidate graphs once, compress runs the
  one you name.
- Error recipes, `LINEAR:MAX_ERROR=V`, `LOG:MAX_ERROR=P%` and
  `SQRT:MAX_ERROR=VN`, shared by compress, profile and bench.
- C entry points `geozl_2d_decompress_c`, `geozl_2d_frame_dsize_c`,
  `geozl_2d_bench_c` and `geozl_2d_grid_c`.
- Six graph terminals where there was one, crossed with eight predictor
  branches. `entropy`, `field_lz`, `zstd`, `transpose>entropy`,
  `transpose>zstd`, `store_lo`.
- Runtime AVX2 dispatch, reported by `geozl.simd_info()`.
- C test suite under `test/`, seven new pytest files, and five new libFuzzer
  harnesses that drive the codecs directly and assert the bound. Coverage went
  from 38% to 97%.
- Documentation site under `docs/`, with a page per codec and a notebook.

### Changed

- The wheel ships `libgeozl` as well as the kernels, so `geozl.compress` works
  from a plain `pip install`.
- x86 release baseline raised to `-march=x86-64-v2`. AVX2 stays a runtime
  choice.
- `-ffp-contract=off` on geozl targets, so a `quant_sqrt` frame rebuilds to the
  same bits on arm64 as on x86.
- Quant encode precomputes the forward map for narrow integer inputs, and
  `quant_linear` decode is vectorized.
- CI caps the ISA by `GEOZL_SIMD` rather than compiler flags, so the sse2 and
  avx2 jobs exercise the two paths of the same binary the wheel ships.

### Removed

- The `"full"` method. There is no brute force sweep on the compress path,
  `geozl.profile` picks the graph instead.
- Python bindings for the pcodec-derived family, `BinOffset`, `IntMult`,
  `FloatQuant` and `FloatMult`. The C codecs stay registered so old frames still
  decode.
- `register_decoders` on `geozl.lossless` and `geozl.lossy`. Use
  `geozl.register_decoders`.
- The root `SPEC.md`. Each codec carries its own `spec.md`.

### Fixed

- Predictor kernels read past the end of a buffer when `width` did not divide
  `nbElts`. The geometry is rejected up front now and the kernels return `int`.
- The `wp_static` trainer kept its histograms in a `static` buffer, which two
  threads compressing at once shared.
- Codec headers were written as native integers, so a frame could not cross
  endianness.
- `nodata` marked only the NaN samples matching the first payload it saw, and
  sent a full mask even when nothing or everything was missing.
- Undefined behaviour in the quantizers found by the new fuzzers, including a
  float to int cast in `quant_log`, the u64 decode table, unclamped warped
  reconstructions, and a parameter set `quant_linear` accepted at one end and
  refused at the other.
- OpenZL errors were swallowed by the bindings instead of propagated.

### Breaking

- `quant_linear` moved from CTid `0x72D780` to `0x72D781`, and its header grew
  from 9 to 10 bytes. `0x72D780` is retired rather than reused, so a 0.7.x lossy
  frame fails to find a decoder instead of being read by the wrong one.
- Lossless frames written on a big-endian host by 0.7.x do not read here.
- `geozl_2d_compress` is replaced by `geozl_2d_compress_c`. It returns an error
  code rather than a `ZL_Report`, `method` is now a full graph recipe name, and
  `max_error` is now an error recipe string. `GEOZL_2D_LOSSLESS` is gone.
- `geozl_node_quant_linear` takes a resolved `quant_linear_params *` instead of
  a `double`.
- `geozl.lossy.QuantLinear` takes a recipe string instead of a float, so
  `QuantLinear(0.5, dtype)` becomes `QuantLinear("LINEAR:MAX_ERROR=0.5", dtype)`.

## [0.7.0] - 2026-07-15

First release.

### Added

- Six lossless spatial predictors, `delta_w` `0x72D701`, `delta_n` `0x72D702`,
  `planar` `0x72D703`, `med` `0x72D705`, `average` `0x72D706` and `wp_static`
  `0x72D707`, plus `deinterleave` `0x72D704` for two-lane interleaved streams.
- Four lossless codecs ported from pcodec, `binoffset` `0x72D708`, `intmult`
  `0x72D709`, `floatquant` `0x72D70A` and `floatmult` `0x72D70B`.
- One near-lossless codec, `quant_linear` `0x72D780`, a uniform grid under an
  absolute error bound.
- `geozl_2d_compress`, which sweeps the predictors with an OpenZL brute force
  selector and keeps the smallest frame.
- Python bindings for every codec, placeable in an `openzl.ext` graph, plus
  `register_decoders` for reading frames back.
- A libFuzzer harness over the decode path.

[Unreleased]: https://github.com/asterisk-labs/geozl/compare/v0.13.1...HEAD
[0.13.1]: https://github.com/asterisk-labs/geozl/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/asterisk-labs/geozl/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/asterisk-labs/geozl/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/asterisk-labs/geozl/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/asterisk-labs/geozl/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/asterisk-labs/geozl/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/asterisk-labs/geozl/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/asterisk-labs/geozl/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/asterisk-labs/geozl/releases/tag/v0.7.0
