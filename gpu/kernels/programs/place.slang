// Copies rectangles of decoded frames into the caller's tensor. Indices and
// strides count ELT_BYTES samples and are checked on the host.

#ifndef ELT_BYTES
#define ELT_BYTES 2
#endif

#if ELT_BYTES == 1
typedef uint8_t Elt;
#elif ELT_BYTES == 2
typedef uint16_t Elt;
#elif ELT_BYTES == 4
typedef uint Elt;
#elif ELT_BYTES == 8
typedef uint2 Elt;
#elif ELT_BYTES == 16
typedef uint4 Elt;
#else
#error "ELT_BYTES must be 1, 2, 4, 8 or 16"
#endif

static const uint THREADS = 128;
static const uint ROWS = 4;

struct place_copy {
  uint source;
  uint sourceRow;
  uint sourceStep;
  uint destination;
  uint destinationRow;
  uint destinationStep;
  uint width;
  uint rows;
};

StructuredBuffer<Elt> source;
StructuredBuffer<place_copy> copies;
RWStructuredBuffer<Elt> target;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void place(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
  place_copy c = copies[group.x];
  uint end = min(c.rows, group.y * ROWS + ROWS);
  for (uint r = group.y * ROWS; r < end; ++r) {
    uint s = c.source + r * c.sourceRow;
    uint d = c.destination + r * c.destinationRow;
    for (uint x = thread.x; x < c.width; x += THREADS)
      target[d + x * c.destinationStep] = source[s + x * c.sourceStep];
  }
}
