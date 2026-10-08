// Helpers shared by the codec benchmarks.

#ifndef GEOZL_GPU_PROTO_BENCH_H
#define GEOZL_GPU_PROTO_BENCH_H

#include "gpu.h"

#include <stddef.h>
#include <stdint.h>

double now_seconds(void);
void *page_alloc(size_t bytes, size_t *allocated);
size_t page_round(size_t bytes);

// Sentinel-2-like 10 m cells, `bands` planes of side x side uint16 each.
void synthesize(uint16_t *raw, size_t cells, size_t bands, size_t side);
// Reads `n` samples from a raw little-endian uint16 file, or synthesizes them
// when path is "synthetic".
uint16_t *load_cells(const char *path, size_t cells, size_t bands, size_t side);

const char *arg(int argc, char **argv, const char *name, const char *def);
int flag(int argc, char **argv, const char *name);

// One frame: a rumi tile of one band, or a whole cell with a plane per band.
typedef struct {
  size_t outOff; // first sample in the output
  size_t nbElts;
  size_t width;
  uint32_t planes;
} Shape;

// Cuts cells into frames, back to back in the returned samples: uint16, or
// with u8 their top 8 of 12 bits.
uint8_t *cut_frames(const uint16_t *raw, size_t cells, size_t bands,
                    size_t side, size_t tile, int wholeCells, int u8,
                    Shape **shapes, size_t *nShapes);

// Best wall time of fn over n items split between threads, or -1 when a call
// returns nonzero.
typedef int (*RangeFn)(void *ctx, size_t begin, size_t end);
double cpu_best(RangeFn fn, void *ctx, size_t n, int threads, int reps);

// Mirrors plane_task in planar_scan.slang.
typedef struct {
  uint32_t outBase;
  uint32_t width;
  uint32_t rows;
} PlaneTask;

typedef struct {
  PlaneTask *tasks; // padded to whole groups with empty planes
  size_t n;
  uint32_t maxWidth, maxRows;
} Planes;

// The fields the GPU planner needs from one PFOR block. `bytes` covers the
// header, exception data and packed body.
typedef struct {
  size_t bytes;
  unsigned exceptions;
} PforBlockInfo;

// Reads and validates one PFOR block header without decoding its values.
// `available` is the number of readable bytes beginning at block.
int pfor_block_info(const uint8_t *block, size_t available, size_t elementBytes,
                    PforBlockInfo *info);

Planes plane_tasks(const Shape *shapes, size_t nShapes);
// Appends the inverse planar dispatches and returns how many.
int scan_dispatches(const Planes *p, GpuDispatch *out);

// Prints the GPU times and compares the output with the source samples;
// returns 0 when they are identical.
int report_gpu(const GpuTimes *gt, const GpuDispatch *d, int nd, double rawMB,
               double planS, const uint8_t *out, const uint8_t *want,
               size_t bytes);

#endif // GEOZL_GPU_PROTO_BENCH_H
