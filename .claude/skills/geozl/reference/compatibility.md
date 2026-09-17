# Frame compatibility and version history

Sources: `docs/compatibility.md`, `docs/c-api.md`, `CHANGELOG.md`,
`bindings/python/test/golden/`.

## Contents

1. The policy
2. Deploying readers and writers
3. Wire-affecting changes by version
4. API changes that break old code
5. Golden frames
6. Changing the wire format of a shipped codec

## 1. The policy

- **0.14.0 is the baseline.** Every release from 0.14.0 on reads frames written by
  0.14.0 and by the compatible releases between 0.14.0 and itself (not newer ones).
- 0.13.x integer lossy frames are outside that promise; a 0.13.x reader refuses a
  0.14.0 integer `quant_linear` frame rather than misreading it.
- **Released CTids are never reassigned.** An incompatible layout gets a new CTid, so an
  old reader fails to find a decoder instead of decoding garbage.
- **Encoders may produce different bytes** in later releases without breaking
  compatibility. Do not compare frame bytes or hashes across versions; compare decoded data.
- **Older readers may not understand newer codecs.**
- GeoZL frames are OpenZL frames; container compatibility follows OpenZL. The Python
  package supports OpenZL 0.2.x (`openzl>=0.2,<0.3`).
- Python API: semantic versioning, may grow before 1.0. C: stable source API from
  0.13.0, no ABI promise before 1.0.

## 2. Deploying readers and writers

- **Upgrade every reader before any writer.** A 0.16.0 writer using a planar recipe emits
  `planar_zigzag` (`0x72D70F`) or `planar_zigzag_pfor` (`0x72D710`); 0.15.x readers do
  not have those decoders.
- Pin the writer version in data products and record it next to the data (for example
  `geozl.__version__` in the dataset metadata) so readers know the minimum version.
- `geozl.compress` output is deterministic for one version, input and graph; that is the
  scope in which byte equality means anything.

## 3. Wire-affecting changes by version

| Version | Change | Consequence |
| --- | --- | --- |
| 0.16.0 (2026-09-01) | new fused codecs `planar_zigzag` and `planar_zigzag_pfor`; `planar>zigzag>...` recipes select them | frames need a 0.16.0 reader; older frames with separate nodes still decode |
| 0.15.1 | `quant_linear` accepts all-zero domains | no format change |
| 0.15.0 | coefficient vectors in the frame header comment | frames without coeffs unchanged |
| 0.14.0 | integer `quant_linear` stores the grid index by default (`STORE=VALUES` keeps the old contents); integer frames need a whole step; `STORE=INDEX` for integer `quant_sqrt`; integer LOG `STORE=INDEX` refused; `pfor` codec added | compatibility baseline; a flag bit that used to be implied is now meaningful, so only the checksum catches its corruption |
| 0.12.0 | predictor headers gain an optional trailing `u32 planes` | 4-byte (13 for `wp_static`) headers still mean one plane |
| 0.11.0 | `nodata` header is the bit pattern only; `all valid` and `all hole` codes removed; empty tiles refused | 0.10.0 nodata frames using those codes no longer decode |
| 0.8.1 | predictor readers refuse row width 0 or wider than the tile | forged or low-level frames with such widths from 0.8.0 are refused |
| 0.8.0 | `quant_linear` moved from `0x72D780` to `0x72D781` with a 10-byte header; codec headers written little endian; new `quant_log`, `quant_sqrt`, `nodata` | 0.7.x lossy frames and big-endian lossless frames do not read |

## 4. API changes that break old code

| Version | Old | New |
| --- | --- | --- |
| 0.14.0 | `store_lo` terminal | removed; `unknown method` |
| 0.10.0 | `geozl.compress(tile, method=..., error=..., nodata=...)` | `g = geozl.graph(tile, method, error=..., nodata=...)`, then `geozl.compress(tile, graph=g)` |
| 0.10.0 | `geozl.decompress(frame, dtype, width)` | `geozl.decompress(frame).view(dtype).reshape(shape)` |
| 0.10.0 | `geozl.profile(..., method=...)` | `geozl.profile(..., prior=...)` |
| 0.10.0 | lossy parameters per tile | frozen by `graph()` from its raster |
| 0.9.0 | `geozl_2d_decompress_c(..., outSize, errCtx, ...)` | `int verify` added before `errCtx` |
| 0.8.0 | `"full"` brute-force method | `geozl.profile` picks the recipe |
| 0.8.0 | `QuantLinear(0.5, dtype)` | `QuantLinear("LINEAR:MAX_ERROR=0.5", dtype)` |
| 0.8.0 | `register_decoders` on `geozl.lossless` and `geozl.lossy` | `geozl.register_decoders` |
| 0.8.0 | Python `BinOffset`, `IntMult`, `FloatQuant`, `FloatMult` | removed (C decoders kept) |
| 0.8.0 | `geozl_2d_compress`, `GEOZL_2D_LOSSLESS` | `geozl_2d_compress_c` with a recipe name and an error string |

Code or docs mentioning `max_error=` as a Python argument, `method=` on `compress`, or
`api-low` pages predate 0.10.0.

## 5. Golden frames

- `bindings/python/test/golden/frames/*.zl` are released frames: 28 written by 0.13.0
  (all six predictors at 1 and 3 planes, `id`, `deinterleave`, `nodata` with an int32
  sentinel and float32 NaN, float quantizers) and 6 by 0.14.0 (integer quantizer frames in
  both storage modes, including `planar>zigzag>pfor`, then written with separate nodes).
  As of 0.16.0 no golden frame uses the fused `0x72D70F` or `0x72D710` CTids.
- `manifest.json` records per frame: `graph`, `error`, `nodata`, `dtype`, `shape`,
  `sha256_frame`, `sha256_decoded`, `frame_bytes` (0.13.0 entries only), and `written_by`
  for frames added after the 0.13.0 set. The manifest is frozen on purpose.
- `test_golden.py` checks the frame hash and the decoded hash; CI runs it on x86-64 and
  on macOS arm64.
- Adding a frame: write it with the release that defines the wire form, add a manifest
  entry with its own `written_by`, never regenerate existing frames.

## 6. Changing the wire format of a shipped codec

Checklist from `docs/compatibility.md`:

1. Freeze a golden frame in every wire form the change touches, including the one being
   retired, before the change lands.
2. Confirm an older reader **refuses** the new frame rather than misreading it. Trace the
   actual predicate; do not assume.
3. Decide whether the change needs a new CTid or only widens which flag values are legal,
   and record the reasoning in `docs/compatibility.md`, not only in the changelog.
4. State in the changelog entry which direction of compatibility survives.
5. Count the redundancy the frame loses: a flag once implied by another field made a bit
   flip detectable; once both values are legal only the checksum remains.

New codecs follow `contributing.md` instead.
