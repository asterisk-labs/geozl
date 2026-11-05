# Preparing GeoZL 0.30.0

This checklist prepares a local release candidate. Pushing a `v*` tag starts
publication, so review the candidate and its CI artifacts first.

## Compatibility

- Keep reading frames written since the 0.20.0 compatibility baseline. Never
  reassign a shipped CTid; preserve golden frames for each wire layout.
- `geozl/gpu.h` joins the supported C source API in 0.30.0. The C ABI is not
  promised before 1.0, so consumers should rebuild after an update.
- Prebuilt Python wheels target Linux x86-64 with CUDA and macOS arm64 with
  Metal. The platform matrix also checks source builds elsewhere.
- The current OpenZL pin is a development commit containing the upstream
  fixes and many other changes since v0.3.0. Review the full submodule diff and
  its compatibility with the Python OpenZL 0.3.x dependency. Before tagging,
  move the pin to the integrated upstream commit, refresh the R vendor copy and
  rerun the validation gates.

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

1. Move the candidate notes into the dated 0.30.0 changelog entry. State the
   compatibility direction of any wire change.
2. Set `VERSION` and run `python tools/sync_r_vendor.py` to synchronize R,
   Julia and npm. `test_release.py` checks their versions.
3. Re-run the validation gates on that exact tree. Review the release
   workflow's dry-run artifacts.
4. Push a `v0.30.0` tag only when publication is intended. The tag workflow
   publishes PyPI wheels and a GitHub release, then npm. R and Julia use their
   own package channels.
