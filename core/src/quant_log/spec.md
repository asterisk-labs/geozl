# quant_log

A relative error bound. Every sample comes back within a fixed fraction of
itself, so a value of 8000 gets the same one per cent of slack as a value of 8.

    LOG:MAX_ERROR=1%                 |x - x^| <= 0.01 * |x|
    LOG:MAX_ERROR=1%,STORE=VALUES    same bound, whole numbers in the stream

The per cent sign is required. `MAX_ERROR=1` is rejected, because it would mean a
bound of one, and nobody means that.

## Why a geometric grid

Ask for the gap between levels to be proportional to the value and you get

    L(q+1) = L(q) * (1 + c)

which is a uniform grid once you take the log. So the levels are powers of a
fixed ratio, `step` is the width of one level in log2, and neighbouring levels
differ by `2^step`. Rounding lands within half a level, so the bound held is
`2^(step/2) - 1`.

Everything is in log2 rather than natural log because the integer part of log2 of
a float is its exponent field, which is exact and free.

## The three cases

The stream is always integer, which is what the codecs behind this one want.
What it holds depends on the input.

| input | recipe | stream holds | decode |
| --- | --- | --- | --- |
| 1. integer | any | the reconstruction | copy the block |
| 2. float | `STORE=VALUES` | the reconstruction | one conversion |
| 3. float | default | the index | one table lookup |

### Case 1, integers

Sentinel-2 DN, Landsat, anything that ships as `uint16`. The stream holds the
snapped value in the original type, so the decoder hands the block straight back.

    LOG:MAX_ERROR=1% on uint16

    in        0     1    50   108   109  1000  8000  65535
    stream    0     1    50   108   108   998  7984  65535
    out       0     1    50   108   108   998  7984  65535
    error     0%    0%    0%    0%  0.92% 0.20% 0.20%    0%

Small values come back exact and it is not a special case in the code. Below 100
the gap between levels is under one, so the level nearest a whole number rounds
back to that whole number. Above it the grid starts to bite. The crossover is at
`0.5 / (sqrt(1+b) - 1)`, which is 100.2 at one per cent, and the first value that
moves is 109.

The largest value comes back exact too, because the top level is folded onto the
type maximum.

### Case 2, floats holding whole numbers

For data that sits away from zero, where a whole number is enough. Temperature in
kelvin, UTM coordinates, radiance in counts.

    LOG:MAX_ERROR=1%,STORE=VALUES on float32

    in      250.00  273.15  290.70  330.40
    stream     250     274     291     331
    out     250.00  274.00  291.00  331.00
    error       0%  0.311%  0.103%  0.182%

The decoder does no arithmetic here, only a conversion from the integer stream to
the output type. That is the fastest case after a plain copy.

It is refused rather than served when the tile reaches below the crossover, since
rounding a small float to a whole number already costs more than the bound.

    LOG:MAX_ERROR=1%,STORE=VALUES on a DEM in metres

    STORE=VALUES rounds to whole numbers, which holds a MAX_ERROR of 1% only
    at or above 100.249, and this tile reaches 0.4

It is also refused when the reconstruction would run past the largest whole
number the float type carries, `2^11`, `2^24` or `2^53`.

    STORE=VALUES needs a reconstruction exact in the output type, and
    3.02202e+07 is past the 1.67772e+07 it carries

Never a quiet fallback to case 3. Whether it fits depends on the tile, so falling
back would make the grid depend on the tile.

### Case 3, floats, the default

Specific humidity, reflectance, anything crossing several decades. The stream
holds the index and the decoder rebuilds the level.

    LOG:MAX_ERROR=1% on float32

    in           0     1e-06    3.4e-04     1.7e-02
    stream       0      4497       4789        4986
    out          0  1.009e-06  3.371e-04   1.700e-02
    error       0%    0.946%     0.853%      0.018%

The same DEM that case 2 refused works here without complaint.

    in         0.0       0.4     120.5    8848.9
    stream       0      5145      5432      5647
    out        0.0       0.4     121.6    8773.9
    error       0%    0.577%    0.931%    0.848%

## Zero

Zero is the one sample no level describes, so it gets index zero and comes back
exact. A NaN has no magnitude to place, lands on the same index and comes back as
zero, so a tile that means its NaNs carries the nodata codec in front. The sign of
a negative zero is not kept.

## The header

Ten bytes, little endian.

    byte 0     element type. 0 to 3 are u8, u16, u32, u64, 4 to 7 the signed
               widths in the same order, then 8 f16, 9 f32, 10 f64
    byte 1     flags. bit 0, the encoder saw no negative sample, so the decoder
               floors at zero. bit 1, the stream holds the reconstruction
    bytes 2-9  step, an IEEE double

Corrupt is a type byte outside 0 to 10, a flag bit above bit 1, bit 1 clear on an
integer type, and a step that is not finite or is at or below zero.

## Where the levels sit

Case 3 anchors at the smallest positive value the type holds, which is a power of
two, so index magnitude one is the bottom and index zero is free for zero.

    x^ = sign(q) * 2^(A + (|q| - 1) * step)      A = -24, -149, -1074

Cases 1 and 2 anchor at one, and the level is rounded to a whole number.

    x^ = sign * round(2^(j * step))

## Cutting the step

Each thing between a sample and its reconstruction is a ratio, so they multiply,
and in log2 that is a sum.

Cases 1 and 2 have two: the grid, and the rounding of the level to a whole
number. Split the budget evenly and each gets `sqrt(1+b)`. That split is also
what sets the crossover, since the whole-number rounding costs `0.5/x` and the
two meet at `0.5 / (sqrt(1+b) - 1)`.

Case 3 has two as well: the grid, and the rounding when the level reaches the
output width, which is half an ulp of the type.

Both add a small term for the two series in `quant_log_math.h`, and a step at or
below zero is a refusal.

    LOG:MAX_ERROR=0.001% on float16

    a MAX_ERROR of 0.001% is at or below what this type rebuilds to, which
    bottoms out near 0.0488281%

## Nothing about the grid comes from the tile

Not the anchor, not the step. The same value rebuilds the same way whatever
raster it was cut from and at whatever chunk size, so a scene compressed at one
tile size and again at another gives the same numbers.

The encoder does read the tile, for three things, and none of them moves a level.
Whether any sample is negative, which sets the floor the decoder clamps against.
The smallest and largest magnitude, which decide the two `STORE=VALUES` refusals.
That is all.

## Where the bound ends

At the smallest normal of a float type. A subnormal `k * 2^E` has neighbours a
flat `2^E` apart, so its relative rounding is `0.5/k` and grows without limit as
`k` falls. No grid reaches that and no constant covers it.

`quant_log_scan` reports a tile that goes below, so a caller can decide.
Everything at or above the smallest normal is covered, checked one value at a
time over the whole of `uint8`, `uint16`, `int16`, `float16` and the normal
`float32`.

    LOG:MAX_ERROR=5%     worst 4.999998e-02   over the bound 0
    LOG:MAX_ERROR=1%     worst 9.999988e-03   over the bound 0
    LOG:MAX_ERROR=0.1%   worst 9.999960e-04   over the bound 0

4261412864 normal float32 values each time.

## Both transcendentals are written out

A frame is written on one machine and read on another, and libm is not correctly
rounded, so two platforms can disagree by an ulp and rebuild the same index two
ways. `log2` and `exp2` live in `quant_log_math.h` in plain IEEE arithmetic, and
with `-ffp-contract=off` they evaluate the same anywhere.

Neither kernel calls libm at all. Only the resolver does, once per tile, and what
it produces is the `step` in the header, which both sides then read.

The forward transform takes sixty-four points across the mantissa, each with its
log2 and its reciprocal, so the mantissa costs a multiply and a short series
instead of a call. Measured against libm across each type's whole range:

    log2  f16   worst 7.1e-15   budget 3.6e-14
    log2  f32   worst 5.7e-14   budget 2.5e-13
    log2  f64   worst 2.3e-13   budget 1.9e-12
    exp2        worst 2.2e-16   budget 1.3e-12

## Speed

The decoder for case 3 builds the levels the tile uses, once, at the output width
and already folded onto the output range, and the table covers the index zero and
both signs so the loop has no branch in it. Above 64 kB of table the long way is
taken instead.

    8M elements, one per cent, on one core

    case 1  uint16   encode   367 MB/s   decode  12648 MB/s
    case 2  float64  encode  1278 MB/s   decode   5820 MB/s
    case 3  float32  encode   649 MB/s   decode   2777 MB/s

## What it buys

The dynamic range and nothing else. Against a flat bound wide enough to hold the
same worst relative error, on the index stream through zstd:

    five decades    1.64x
    one decade      1.24x
    a narrow band   1.03x

Over a narrow band a relative bound is nearly a flat one, and there is nothing to
win.
