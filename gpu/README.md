# GPU decode

Decode-only implementation for the PFOR- and PivCo-backed lossless graphs. It
supports 1- and 2-byte elements without changing their frame format, and
`core/include/geozl/gpu.h` exposes it for readers such as Rumi.

The kernels are written in [Slang](https://shader-slang.org) and compiled to
Metal source or CUDA PTX. OpenZL checks the outer frame and its compressed
checksum on the CPU. The capture layer intercepts the supported codec inputs
and stops before the raster is materialized; the host planner then validates
the PFOR or PivCo streams and builds the GPU task lists. Planning splits a
batch into at most four contiguous frame ranges and merges the task lists in
input order.

## Layout

| Path | What it is |
|---|---|
| `kernels/programs/pfor.slang` | PFOR-backed decoder program |
| `kernels/programs/pivco.slang` | PivCo-backed decoder program |
| `kernels/programs/place.slang` | copies decoded rectangles into the caller's memory |
| `../core/src/gpu/modules/` | compiled programs the library embeds, from `tools/build_gpu_modules.py` |
| `../core/src/gpu/decode.c` | the `geozl/gpu.h` batch decoder |
| `kernels/stages/` | PFOR, PivCo and inverse planar/Zigzag stages |
| `kernels/common/` | shared payload access and constants |
| `../core/src/gpu/` | frame capture, GPU task planning and the backend runtimes |
| `../test/gpu/` | host-side frame preparation tests |
| `proto/gpu.c`, `gpu.h` | benchmark timing adapter |
| `proto/pfor_bench.c`, `proto/pivco_bench.c` | CPU/GPU benchmarks and verification |
| `proto/run_all.sh` | benchmark driver |

PFOR assigns one wave to each 256-value block. Inverse planar uses one pass for
planes up to 256 columns and separate row and column passes for wider planes.

PivCo rebuilds each Huffman block from the leaves to the root. Each lane of a
wave consumes one bitmap byte at a time and merges eight symbols; alternate tree levels use
the lane and output allocations as ping-pong storage. It then decodes any PFOR
byte lanes, joins the lanes and runs inverse planar.

The runtime keeps compiled kernels in a program and device allocations in a
batch. A device-only output marked `GEOZL_GPU_RESULT` can be moved into an
owning `GeozlGpuMemory`. The result remains valid after its batch and program
are destroyed; copying it back to the host is a separate operation.

`runtime.c` validates every call, keeps the shared batch state and forwards to
the backend chosen by `GeozlGpuBackendId`. `runtime_metal.m` and
`runtime_cuda.c` hold only device code behind a `GeozlGpuBackendOps` table, and
each compiles in under `GEOZL_GPU_METAL` or `GEOZL_GPU_CUDA`. A Vulkan or HIP
backend adds one such file, an id and a Slang target; capture, planning and the
decoders stay the same. Each backend also provides staging: page-locked host
memory uploaded to CUDA without another host copy, and a shared Metal buffer
the device reads directly. The kernels assume 32-wide waves, so a backend must
provide them or reject the device, as Metal does.

A program may run only one batch at a time. CUDA stores its buffer pointers in
module globals, so callers that need concurrent work must create one decoder
per stream. Decoder creation also compiles the generated Metal source or JITs
the PTX; reuse a decoder instead of creating one for every batch.

The driver JITs PTX only up to its own PTX version, so build PTX with an NVRTC no
newer than the oldest driver to support. PTX from NVRTC 12.8 (`.version 8.7`)
fails on a 550 driver, which accepts 8.4; the runtime error carries the JIT log.

`geozl_pfor_gpu_decode()` and `geozl_pivco_gpu_decode()` are complete paths over
that runtime. They take complete GeoZL frames, check and plan them on the CPU,
and return one contiguous `GeozlGpuMemory` in frame order. The GPU path
recognizes `id>pfor`, `planar>zigzag>pfor`, `id>pivco`,
`id>transpose>pivco` and the fused `planar>zigzag>transpose>pivco` codec. PivCo
capture is keyed by the fixed GeoZL CTids `0x72D713` and `0x72D712`; it does not
depend on OpenZL's choice of standard entropy nodes. The planar codec already
splits each residual element into byte lanes internally.

Capture and planning write the kernel inputs into staging that the decoder
keeps between calls, so a run uploads them without a host copy. The first call
sizes that staging. It stays at the largest batch seen, including the capture
bound of frame plus output bytes, until the decoder is destroyed.

If OpenZL stores a frame raw because compression does not make it smaller,
capture decodes that frame on the CPU and copies it into the device result
without changing the rest of the batch. These lower-level PFOR and PivCo
decoders take a maximum output size, checked before host or device allocation.

The compressed-frame checksum is checked before capture. The content checksum
is not checked because the reconstructed samples remain on the device.

## Decoding into device memory

`geozl_gpu_decode` takes compressed frames, the bytes each decodes to, and
copies of strided sample rectangles into a CUDA pointer or Metal buffer. A
frame goes to the device when the last codec in its header is PFOR or PivCo
and its elements match the sample size, which is one or two bytes. Every other
frame, including quantized, NoData and zstd frames, decodes on the host with
OpenZL into page-locked staging. A valid device frame that needs another host
transform falls back the same way; malformed codec metadata rejects the call.
Host threads capture and decode frames from a shared counter; planning then
splits each codec's frames over the same threads.

The decoder is created with a nonzero limit for the sum of decoded frame bytes
in one call. Copies may interleave samples, but no two copies may write the
same target sample. On CUDA the target allocation stays alive on the selected
device until later work in the same stream, synchronization, the next decode
or decoder destruction has completed.

Device frames decode into scratch the decoder keeps, and `place` copies every
requested rectangle into the target. On CUDA the work joins the caller's
stream and the call returns once it is queued. The next call, synchronization
or destruction waits for it before reusing staging. Metal finishes each batch
before returning. Programs keep finished device allocations for the next batch,
so steady calls neither allocate nor free device memory.

The kernels ship in the library as PTX and Metal source, embedded at build
time from `core/src/gpu/modules`. Regenerate them with
`tools/build_gpu_modules.py` after a kernel change: Metal source anywhere
slangc runs, PTX with the NVRTC of the oldest CUDA to support. The checked-in
PTX comes from NVRTC 12.2. The script records the compiler inputs and hashes of
every generated module in `modules/sources.sha256`; CI checks both the sources
and the embedded outputs.

## Running

```sh
cd proto
make metal SLANGC=/path/to/slangc      # macOS, Slang 2026.19
make cuda SLANGC=/path/to/slangc       # Linux, Slang 2026.19
make hosts-cuda                        # driver only, PTX built elsewhere
make check                             # capture, planner and validation, no GPU
make check-runtime-metal               # Metal runtime and staging
make check-pfor-metal SLANGC=...       # full PFOR decode and wire edge cases
make check-pivco-metal SLANGC=...      # PivCo block-tail regression
make check-cuda SLANGC=...             # the same decode checks on CUDA
make check-gpu-decode                  # geozl/gpu.h with mixed codecs and copies
BACKEND=metal ./run_all.sh
```

The PivCo benchmark builds a static GeoZL in `core/build-gpu`.
Shader builds check Slang's reflected buffer order against the host bindings.
`--input synthetic` generates test cells. `REAL` may point to raw little-endian
uint16 data with shape `N x 4 x 1056 x 1056`. `GPU=n` selects a CUDA device.

`kernel` measures dispatch execution. `+ transfer` runs from input submission
until the output is ready on the device. Planning, resource creation and the
final verification copy are excluded.
