# Preparing GeoZL 0.32.0

This checklist prepares a local release candidate. Pushing a `v*` tag starts
publication, so review the candidate and its CI artifacts first.

## Compatibility

- Never reassign a shipped CTid.
- The golden corpus holds one frame per wire layout the current readers decode.
  Rerun `python tools/write_golden.py` after any wire change, so every frame is
  written by this release.
- The C source API and ABI are not promised before 1.0, so consumers should
  rebuild after an update.
- Prebuilt Python wheels target Linux x86-64 with CUDA and macOS arm64 with
  Metal. The platform matrix also checks source builds elsewhere.
- The OpenZL pin is the integrated `dev` commit `b7f498b`. Check frame
  compatibility with the Python OpenZL 0.3.x dependency before changing it.

## Validate the candidate

Run these against the exact source that would be tagged:

```bash
make test-san
make -C gpu/proto check CMAKE_FLAGS=-DGEOZL_WERROR=ON
make fuzz-replay
ruff check .
mypy
make r-check
make julia
make wasm-test
python tools/build_gpu_modules.py --check
```

Run `make -C gpu/proto check-metal` and `check-cuda` on machines with those
devices. Require `Status: OK` from `make r-check`. Run the CI platform matrix
for Linux arm64, macOS x86-64 and Windows.

Build the Python wheels and npm package, inspect them with
`bindings/python/check_wheel.py` and `npm pack`, then install and smoke-test the
artifacts in clean environments.

## Version and publication

1. Move the candidate notes into the dated 0.32.0 changelog entry. State the
   compatibility direction of any wire change.
2. Set `VERSION` and run `python tools/sync_r_vendor.py` to synchronize R,
   Julia and npm. `test_release.py` checks their versions.
3. Re-run the validation gates on that exact tree. Review the release
   workflow's dry-run artifacts.
4. Push a `v0.32.0` tag only when publication is intended. The tag workflow
   publishes PyPI wheels and a GitHub release, then npm. R and Julia use their
   own package channels.
