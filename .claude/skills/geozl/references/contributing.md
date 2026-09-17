# Working on the geozl repository

For changes inside `github.com/asterisk-labs/geozl`. Sources: `Makefile`,
`core/CMakeLists.txt`, `docs/adding-a-codec.md`, `docs/README.md`, `.github/workflows/`,
the test suites.

## Contents

1. Repository map
2. Setup, build and test commands
3. Test suites and the guards that matter
4. Adding a codec
5. Where validation belongs
6. The Python side of a codec
7. Documentation site rules
8. Conventions
9. CI, fuzzing and releases

## 1. Repository map

```text
core/
  include/geozl/        public headers: geozl.h, ctids.h, dtype.h, coeffs.h, export.h, quant_*_params.h
  src/<codec>/          encode_<codec>_{kernel,binding}.{c,h}, decode_..., spec.md
  src/2d/2d.c           recipe grammar, graph builder, geozl_2d_* entry points, bench
  src/lossy/            recipe parse, SQRT fit, resolve, domain check
  src/common/           simd.c/h, scan.h, vprefix.h, vshift.h, raster.h, endian.h, half.h, recipe_parse.h, graph_num1to{1,2}.h
  src/encoder_registry.c  geozl_node_* builders
  src/decoder_registry.c  kDecoders table, geozl_register_decoders
  CMakeLists.txt        codec discovery by folder convention
bindings/python/
  geozl/_ffi.py         hand-written cffi cdef (ABI mode) and library loading
  geozl/_2d.py          profile, graph, compress, decompress, error shorthand
  geozl/_codec.py       spatial_predictor and quantizer factories
  geozl/_dtype.py _coeffs.py _simd.py
  geozl/lossless/ lossy/  one module per exposed codec
  test/                 pytest suites, golden/ frames and manifest
  pyproject.toml        hatchling, version from ../../VERSION
test/*.c                standalone C kernel tests (no OpenZL)
fuzz/                   libFuzzer harnesses, replay/ inputs kept in git
docs/                   static site (no build step), compatibility.md, c-api.md, adding-a-codec.md, notebooks/
extern/openzl           OpenZL submodule (v0.2.0)
licenses/ NOTICE        vendored notices shipped in wheels
VERSION CHANGELOG.md    single version source and Keep a Changelog history
```

## 2. Setup, build and test commands

Requirements: Python 3.11+, a C11 compiler, Git, Make, CMake, Ninja.

```bash
python -m pip install cmake ninja numpy cffi openzl pytest ruff mypy
make submodules                 # git submodule update --init --recursive
make python FULL=ON             # build, stage libs into bindings/python/geozl/_lib, pip install -e
make test                       # test-c, then pytest bindings/python
ruff check .
mypy
```

| Target | What it does |
| --- | --- |
| `make` or `make python` | build, stage libraries, editable install, import smoke test |
| `make build` | CMake build only (`core/build`) |
| `make lib` | build and copy `libgeozl_kernels` and `libgeozl` next to the binding |
| `make test-c` | compile every `test/test_*.c` against all kernels and run them (no CMake, no OpenZL) |
| `make test` | `test-c` plus pytest |
| `make test-san` | both suites under ASan and UBSan in `core/build-san` |
| `make exhaustive` | C tests over every value of a type (`GEOZL_EXHAUSTIVE=1`, minutes) |
| `make fuzz`, `fuzz-check`, `fuzz-replay`, `fuzz-build`, `clean-fuzz` | libFuzzer runs, CI-style check, replay of `fuzz/replay` and cached corpora |
| `make install PREFIX=/opt/x` | `cmake --install` |
| `make clean` | remove builds, staged libraries, caches |

Variables: `BUILD=Debug`, `PYTHON=python3.12`, `GEN='Unix Makefiles'`, `FULL=OFF`
(kernels only: no `libgeozl`, so `geozl.compress` and every high-level test are
unavailable), `SAN=ON`, `FUZZ_TIME=600`, `FUZZ_JOBS=8`,
`CMAKE_FLAGS=-DGEOZL_USE_SYSTEM_OPENZL=ON`, `CLANG=/path/to/clang` (on macOS the fuzzers
need Homebrew LLVM).

Single tests:

```bash
python -m pytest bindings/python/test/test_2d.py::test_error_bound_holds -q
make test-c                                   # all C tests; binaries land in core/build/ctest/
```

After editing C sources, rerun `make python FULL=ON` (or `make lib`) before pytest,
otherwise Python keeps loading the previously staged libraries.

## 3. Test suites and the guards that matter

| File | Guards |
| --- | --- |
| `test_2d.py`, `test_graph.py` | high-level API contract: shorthand errors, grid sizes per width, profile bytes equal compress bytes, lossy domains, nodata surface |
| `test_codecs.py` | every predictor bit exact across 11 dtypes, shapes and patterns; quantizer bounds; checksum refusal |
| `test_cross_reader.py` | **the strongest check**: C-written frames decode with Python decoders and the reverse, byte for byte on lossy frames |
| `test_cdef.py` | the `_CDEF` block in `_ffi.py` matches the C headers (ABI mode verifies nothing; drift corrupts memory silently) |
| `test_golden.py` | frozen released frames still decode to the recorded hashes |
| `test_malformed.py`, `test_headers.py` | decoders refuse forged headers and streams |
| `test_catalog.py` | the codec catalog agrees across `ctids.h`, README, `docs/docs.html`, `docs/assets/js/main.js`, `docs/codecs/*.html` |
| `test_docs.py` | `docs/api-high.html` signatures equal the source signatures; notebook mentions `profile`, `graph`, `compress`, `decompress`; no `max_error` or `api-low` in maintained docs; local links resolve |
| `test_release.py` | `VERSION` has a `## [X.Y.Z] - date` entry and compare links in `CHANGELOG.md` |
| `test_licenses.py` | `licenses/` copies equal the vendored upstream texts and `NOTICE` lists them |
| `test_simd.py`, `test_ffi.py` | SIMD capping via `GEOZL_SIMD` in fresh processes; library lookup errors |
| `test_planes.py`, `test_nodata.py`, `test_noise.py`, `test_categorical.py`, `test_coeffs.py`, `test_wpstatic.py`, `test_deinterleave.py`, `test_planar_zigzag_pfor.py`, `test_linear_zero.py` | feature suites |

C tests (`test/`): per-kernel round trips (`test_pfor.c`, `test_planar_zigzag*.c`,
`test_nodata.c`, `test_planes.c`), quantizer bound walks (`test_quant_*.c`), dtype table
agreement, header endianness, coefficient blobs, multithreaded `wp_static` training.

Test through the buffer or mask a kernel produces, not a value it returns, or the test
only proves the function agrees with itself.

## 4. Adding a codec

A codec is a self-contained folder `core/src/<name>/`, shaped like an OpenZL codec so it
could move upstream. File names are load bearing: CMake globs
`src/<name>/encode_<name>_binding.c` and derives the rest.

Checklist (from `docs/adding-a-codec.md`):

- [ ] Four file pairs: `encode_<name>_kernel.{c,h}`, `decode_<name>_kernel.{c,h}` (pure C,
      no OpenZL, dispatch by element width), `encode_<name>_binding.{c,h}`,
      `decode_<name>_binding.{c,h}` (typed OpenZL encoder and decoder). Reuse
      `GEOZL_NUM1TO1_GRAPH` or `GEOZL_NUM1TO2_GRAPH` from `common/` when they describe the
      streams; add `graph_<name>.h` when the stream types are the codec's own (the three
      quantizers carry one because their output is always an integer stream, and serial
      outputs such as `pfor` need their own).
- [ ] Binding headers declare `EI_geozl_<name>` / `DI_geozl_<name>` and the descriptor
      macros `EI_<NAME>(id)` / `DI_<NAME>(id)`.
- [ ] CTid in `core/include/geozl/ctids.h`: next free id in the right band, lossless
      `0x72D700` to `0x72D77F`, lossy `0x72D780` to `0x72D7FF`. Never reuse or renumber.
- [ ] Decoder row `REGISTER(GEOZL_CTID_<NAME>, DI_<NAME>)` in `decoder_registry.c`.
- [ ] Node builder in `encoder_registry.c`, declared in `geozl.h` with `GEOZL_API`
      (a width predictor is one line through `geometry_node` or `width_node`); refuse bad
      caller input at the top.
- [ ] `core/CMakeLists.txt` only for sources outside the naming convention (after the
      loop, next to the `wp_static` trainer, spec parsers, SQRT fit and `common/simd.c`).
- [ ] `spec.md`: enough for a reader to invert the codec without the encoder
      (`### Inputs`, `### Codec Header`, `### Decoding`, `### Outputs` for simple codecs).
- [ ] Codec header: the only channel from encoder to decoder. Write it with
      `ZL_Encoder_sendCodecHeader`, read with `ZL_Decoder_getCodecHeader`, keep it minimal,
      little endian (`common/endian.h`).
- [ ] If exposed to Python: `_CDEF` entries, a module under `lossless/` or `lossy/`,
      exports in the package `__init__.py`, and the decoder in `_DECODERS`.
- [ ] Catalog in all five places or `test_catalog.py` fails (section 7).
- [ ] Round trips in C across every element width and edge shapes, bit exact for
      lossless and within the bound for lossy; add the codec to `test_cross_reader.py`.
- [ ] `CHANGELOG.md` entry under `## [Unreleased]`.

Syntax-check a binding against the real OpenZL headers:

```bash
gcc -fsyntax-only -std=c11 -Icore/include -Icore/src -Iextern/openzl/include \
    core/src/<name>/decode_<name>_binding.c
```

To add the codec to the recipe grid instead of (or besides) exposing a node, edit
`core/src/2d/2d.c`: the predictor or terminal enum, its name function (parse and format
share one spelling), `build_candidate`, and the width filter in `geozl_2d_grid_c`. Then
update grid-size tests in `test_2d.py`.

## 5. Where validation belongs

- Everything a **kernel** computes on is checked by the kernel: shifts, bin widths, dtype
  codes used as table indices, unknown element widths. Kernels are exported from
  `libgeozl_kernels` and reachable from Python without any frame or binding. A refused
  call still leaves its output buffer readable.
- The **decode binding** checks what only it sees: header length, field layout, and the
  relation between header values and the stream widths OpenZL reports; it rejects with
  `ZL_ErrorCode_corruption`.
- Codecs with several preconditions share a `<name>_check.h` between both ends (the
  quantizers' `quant_<name>_check.h`, `pfor/pfor_check.h`).
- Anything read from the wire that drives an allocation is bounded by the payload size
  first (PFOR: at most 128 elements per payload byte).

## 6. The Python side of a codec

Python reuses the C kernels and reimplements only the OpenZL encoder, decoder and header
packing, so the cross-reader test can compare two independent bindings.

- Spatial predictor: one call in a four-line module.

  ```python
  from .._codec import spatial_predictor

  Planar, PlanarDecoder = spatial_predictor(
      0x72D703, "geozl.lossless.planar", "planar_encode", "planar_decode")
  ```

  Pass `supports_planes=False` for codecs that only read left.
- Quantizer: `quantizer(ctid, name, prefix, doubles, no_grid_message, extra=())`.
- Anything else: write `CustomEncoder` and `CustomDecoder` subclasses (see
  `lossless/nodata.py`, `lossless/pfor.py`) and repeat every check the C binding makes.
- A kernel signature change is two edits, header and `_CDEF`, never one.

## 7. Documentation site rules

- `docs/` is hand-written static HTML: one shared CSS file, optional JavaScript, pages
  open from disk. Preview with `python3 -m http.server 8000 --directory docs`.
- The codec catalog appears five times: `ctids.h` (checked one way, a codec may exist in
  C before it is documented), the README codec table, the cards in `docs/docs.html`, the
  `CODECS` array in `docs/assets/js/main.js`, and a codec page whose file name uses
  hyphens (`docs/codecs/delta-n.html`, `docs/codecs/planar-zigzag-pfor.html`). Codec
  families carry colours: purple predictors, green stream splitters, orange near lossless.
- Signatures shown on `docs/api-high.html` must match `_2d.py` and `_coeffs.py` exactly.
- Each page marks its own nav entry with `class="here"` in the HTML.

## 8. Conventions

- C11, `-Wall -Wextra` on geozl targets (`GEOZL_WERROR=ON` in CI), hidden visibility with
  `GEOZL_API` on public symbols, `-ffp-contract=off` everywhere so reconstructions are
  bit identical across CPUs.
- Wire integers are little endian; never write native structs.
- Kernels return `int` (nonzero refuses the geometry) unless nothing can be refused.
- SIMD: runtime dispatch through `geozl_simd_has`; AVX2 via target attributes so x86
  wheels keep the base ISA; `-DGEOZL_NO_SIMD` forces scalar; frames must be identical on
  every path.
- Python: ruff (line length 100, target py311, rules `E F I B UP NPY`), mypy with
  `disallow_untyped_defs` on the public modules; cffi plumbing modules are relaxed.
- Comments explain why in full sentences; docstrings are short.
- Commit subjects in this repository are short, lowercase and imperative, for example
  `add fused planar Zigzag codecs`, `fix FULL=OFF planar PFOR tests`.
- `CHANGELOG.md` follows Keep a Changelog with `Added`, `Changed`, `Fixed`, `Removed`,
  `Breaking`; wire-format entries say which compatibility direction survives.
- `VERSION` is the only version source; `.githooks/pre-commit` validates its format.

## 9. CI, fuzzing and releases

CI (`.github/workflows/ci.yml`) on every push and pull request:

- Tests on Python 3.11 to 3.14 with `make python FULL=ON`; ruff and mypy on 3.12; Python
  coverage badge committed on `main`.
- Golden frame decode on macOS arm64.
- Sanitizers with `FULL=OFF` and with the full library (the latter covers `2d.c`).
- Strict build with warnings as errors.
- ISA matrix: scalar (`-DGEOZL_NO_SIMD`), sse2 and avx2 (runtime cap `GEOZL_SIMD`), neon (macOS).

Fuzzing (`fuzz.yml`): replay of `fuzz/replay` on relevant pushes and pull requests; nightly
`make fuzz-check` at 420 seconds per target with a cached corpus. Targets: `decode`,
`binding` (valid frames around fuzzed headers), `roundtrip`, `lossy_recipe`,
`quant_linear`, `quant_log`, `quant_sqrt`, `pfor`, `coeffs`.

Releases (`release.yml`) on tags `v*`: the tag must equal `VERSION`; wheels for
manylinux 2.28 x86-64 and macOS 11 arm64 are built, repaired, smoke tested (import, SIMD
report, profile, graph, compress, decompress round trip) and published to PyPI with
trusted publishing, plus a GitHub release. Record the release in `CHANGELOG.md` and
`VERSION` first.

Security: malformed frames that crash, hang or exhaust resources are vulnerabilities.
Report privately to `hello@asterisk.coop` with `[geozl security]` in the subject.
