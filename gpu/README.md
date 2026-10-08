# GPU decode prototype

Experimental decode-only implementation for `planar_zigzag_pfor` (`0x72D710`)
and `planar_zigzag_pivco` (`0x72D712`). It supports 1- and 2-byte elements.

The kernels are written in [Slang](https://shader-slang.org) and compiled to
Metal source or CUDA PTX. The CPU decoder validates each frame before the host
builds the GPU task lists.

## Layout

| Path | What it is |
|---|---|
| `kernels/planar_zigzag_pfor.slang` | `planar_zigzag_pfor` decoder |
| `kernels/planar_zigzag_pivco.slang` | `planar_zigzag_pivco` decoder |
| `kernels/payload.slang`, `pfor_block.slang`, `pivco_lanes.slang`, `planar_scan.slang` | shared kernels |
| `proto/gpu.h`, `gpu_metal.m`, `gpu_cuda.c` | Metal and CUDA host code |
| `proto/pfor_bench.c`, `proto/pivco_bench.c` | CPU/GPU benchmarks and verification |
| `proto/pivco_plan.c` | PivCo task-list builder |
| `proto/run_all.sh` | benchmark driver |

PFOR assigns one wave to each 256-value block. Inverse planar uses one pass for
planes up to 256 columns and separate row and column passes for wider planes.

PivCo builds a rank directory for the node bitmaps, decodes the Huffman and
PFOR byte lanes, joins the lanes, then runs inverse planar.

## Running

```sh
cd proto
make metal SLANGC=/path/to/slangc      # macOS
make cuda SLANGC=/path/to/slangc       # Linux
make hosts-cuda                        # driver only, PTX built elsewhere
make check                             # host-side planner tests
BACKEND=metal ./run_all.sh
```

The PivCo benchmark builds a static GeoZL in `core/build-gpu`.
`--input synthetic` generates test cells. `REAL` may point to raw little-endian
uint16 data with shape `N x 4 x 1056 x 1056`. `GPU=n` selects a CUDA device.

`kernel` measures dispatch execution. `+ transfer` runs from input submission
until the output is ready on the device. Planning, resource creation and the
final verification copy are excluded.
