# Frame compatibility

Sources: `docs/compatibility.md`, `docs/c-api.md`, `bindings/python/test/golden/`.

## Contents

1. The policy
2. Deploying readers and writers
3. Golden frames
4. Changing the wire format of a shipped codec

## 1. The policy

- **Nothing is stable before 1.0.** A release may change a codec layout, and its readers
  may refuse frames earlier releases wrote. Superseded layouts, names and aliases are
  deleted, not kept behind a shim.
- **Released CTids are never reassigned.** An incompatible layout gets a new CTid, so an
  old reader fails to find a decoder instead of decoding garbage. Retired CTids
  (`0x72D708` to `0x72D70B`, `0x72D70E`, `0x72D780`) stay reserved in `ctids.h`.
- **Encoders may produce different bytes** in later releases. Do not compare frame bytes or
  hashes across versions; compare decoded data.
- **Older readers may not understand newer codecs.**
- GeoZL frames are OpenZL frames; container compatibility follows OpenZL. The Python
  package supports OpenZL 0.3.x (`openzl>=0.3,<0.4`). Every recipe writes OpenZL format 27,
  so frames need an OpenZL 0.3 reader.
- The Python API, the C source API (`geozl/geozl.h`, `geozl/gpu.h`) and the C ABI may change
  in any release before 1.0. Rebuild C applications after an update.

## 2. Deploying readers and writers

- **Upgrade every reader before any writer.** A newer writer may emit a CTid an older
  reader has no decoder for.
- Pin the writer version in data products and record it next to the data (for example
  `geozl.__version__` in the dataset metadata) so readers know the minimum version.
- `geozl.compress` output is deterministic for one version, input and graph; that is the
  scope in which byte equality means anything.

## 3. Golden frames

- `bindings/python/test/golden/frames/*.zl` hold one frame per wire layout the current
  readers decode: `planar_zigzag_lanes` (`0x72D712`), `lanes`
  (`0x72D713`) at 1, 2, 4 and 8 bytes and with Huffman lanes, `planar_zigzag_pfor`
  (`0x72D710`) and `med_zigzag` (`0x72D711`).
- `manifest.json` records per frame: `graph`, `error`, `nodata`, `dtype`, `shape`,
  `written_by`, `sha256_frame` and `sha256_decoded`.
- `test_golden.py` checks the frame hash and the decoded hash; CI runs it on x86-64 and
  on macOS arm64. The R, Julia and WASM tests decode the same corpus.
- Adding a frame: write it with the release that defines the wire form, add a manifest
  entry with its own `written_by`, never regenerate existing frames. When a layout is
  removed, delete its frames and manifest entries with it.

## 4. Changing the wire format of a shipped codec

Checklist from `docs/compatibility.md`:

1. Freeze a golden frame in every wire form the change touches before the change lands.
2. Confirm an older reader **refuses** the new frame rather than misreading it.
3. Decide whether the change needs a new CTid or only widens which flag values are legal,
   and record the reasoning in `docs/compatibility.md`, not only in the changelog.
4. State in the changelog entry which direction of compatibility survives.
5. Count the redundancy the frame loses: a flag once implied by another field made a bit
   flip detectable; once both values are legal only the checksum remains.

New codecs follow `contributing.md` instead.
