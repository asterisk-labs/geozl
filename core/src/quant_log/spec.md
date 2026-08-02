# quant_log

A relative error bound. Every sample comes back within a fixed fraction of
itself, so a value of 8000 gets the same one per cent of room as a value of 8.

    LOG:MAX_ERROR=1%                 |x - x^| <= 0.01 * |x|
    LOG:MAX_ERROR=1%,STORE=VALUES    same bound, whole numbers in the stream
    LOG:MAX_ERROR=0.5%,STORE=INDEX   the default, written out

That is the whole grammar. `MAX_ERROR` is required and has to end in a per cent
sign, because `MAX_ERROR=1` would mean a bound of one and nobody means that. The
value has to land inside `(0, 100)`. `STORE` defaults to `INDEX`. An unknown key,
a key given twice and a trailing comma are all refused.

The sibling codec `quant_linear` gives a flat bound instead. Reach for that one
when the data sits in a narrow band and for this one when it crosses decades.

## The grid

A flat bound wants the levels an equal distance apart. A relative bound wants the
distance to grow with the value, so the levels sit at an equal ratio.

    L(q+1) = L(q) * (1 + c)

Take the log and a constant ratio turns back into a constant gap, so the grid is
uniform again and quantising is a divide and a round. `step` is that gap.
Neighbouring levels are a factor `2^step` apart, rounding lands inside half of
it, and so the grid on its own holds a bound of `2^(step/2) - 1`.

log2 rather than natural log, because the integer part of log2 of a float is its
exponent field. It is already there and it is exact. Only the mantissa needs a
series.

## The three cases

The stream is always an integer type, which is what the codecs behind this one
want. What it holds depends on what came in.

| input | recipe | stream holds | decode |
| --- | --- | --- | --- |
| 1. integer | any | the reconstruction | copy the block |
| 2. float | `STORE=VALUES` | the reconstruction | one conversion |
| 3. float | default | the index | one table lookup |

Cases 1 and 2 share a grid anchored at one, with every level rounded to a whole
number.

    x^ = sign * round(2^(j * step))

Case 3 anchors at the smallest positive value the type can hold. Index magnitude
one is the bottom rung, which leaves index zero free to mean zero.

    x^ = sign(q) * 2^(A + (|q| - 1) * step)      A = -24, -149, -1074

### Case 1, integers

Sentinel-2 DN, Landsat, anything that ships as uint16.

    LOG:MAX_ERROR=1% on uint16

    in        0     1    50   108   109  1000  8000  65535
    stream    0     1    50   108   108   998  7984  65535
    out       0     1    50   108   108   998  7984  65535
    error     0%    0%    0%    0%  0.92% 0.20% 0.20%    0%

The decoder hands the block straight back, so this is the cheapest of the three.
The top level is folded onto the largest magnitude the type holds, which is why
65535 survives here. A signed type reaches one step further below zero than
above it, so that magnitude is 32768 on an int16 and the store cuts the positive
side back to 32767. Fold onto the maximum instead and the most negative value of
the type comes back one short, which a bound tight enough is over.

### Case 2, floats holding whole numbers

For data that sits well away from zero, where a whole number is enough.
Temperature in kelvin, UTM coordinates, radiance in counts.

    LOG:MAX_ERROR=1%,STORE=VALUES on float32

    in      250.00  273.15  290.70  330.40
    stream     250     274     291     331
    out     250.00  274.00  291.00  331.00
    error       0%  0.311%  0.103%  0.182%

The decoder does no arithmetic at all here, only a conversion from the integer
stream to the output type.

### Case 3, floats, the default

Specific humidity, reflectance, anything crossing several decades. The stream
holds the index and the decoder rebuilds the level from it.

    LOG:MAX_ERROR=1% on float32

    in           0     1e-06    3.4e-04     1.7e-02
    stream       0      4497       4789        4986
    out          0  1.009e-06  3.371e-04   1.700e-02
    error       0%    0.946%     0.853%      0.018%

A DEM in metres works here too, and it is the same tile case 2 turns down.

    in         0.0       0.4     120.5    8848.9
    stream       0      5145      5432      5647
    out        0.0       0.4     121.6    8773.9
    error       0%    0.577%    0.931%    0.848%

## Cutting the step

Work backwards from the bound. Everything standing between a sample and its
reconstruction is a ratio, and ratios multiply, so in log2 they add. Take the
budget `log2(1+b)`, subtract what each piece of machinery costs, and double what
survives. That is `step`.

Case 3 pays for two pieces.

    the grid           half a step, which is the thing being solved for
    the output float   half an ulp of the type, so 2^-11, 2^-24 or 2^-53

Cases 1 and 2 pay for two others.

    the grid           half a step
    the whole number   the level is rounded to an integer

The second one is awkward, because it is not a fixed ratio. Rounding x to the
nearest whole number costs `0.5/x`, which gets cheaper as x grows. The codec
splits the budget evenly and hands each side `sqrt(1+b)`. The next section is why
that split is the only one that works.

Both cases also give up a little to the two series in `quant_log_math.h`, once
for the log2 the encoder takes and once for the exp2 the decoder runs. Those
terms sit around 1e-12 and decide nothing until the bound is tight enough to be
refused anyway.

    LOG:MAX_ERROR=0.001% on float16

    a MAX_ERROR of 0.001% is at or below what this type rebuilds to, which
    bottoms out near 0.0488281%

## Where small values come back exact

Call the grid's share `g = sqrt(1+b) - 1`. Near a value x the levels sit about
`2*x*g` apart, so the nearest one is within `x*g` of x. Below

    x = 0.5 / g

that distance is under half a unit, the level nearest a whole number rounds
straight back to that whole number, and the reconstruction is exact. An integer
sample is already whole and rides this for free. There is no special case for it
anywhere in the code, and no constant for the crossing either. It falls out of
the split.

At one per cent `g` is 0.004988 and the crossing is 100.25. Walk every uint16 at
that bound and nothing below 109 moves.

The crossing is also the argument for the even split. Below it the whole-number
rounding is free. Above it the rounding costs `0.5/x`, which is smaller than `g`.
The two are equal at the crossing itself, so that is where the worst case lives,
and it comes to `(1+g)^2`, which is `1+b` exactly. Hand the grid a bigger share
and the crossing slides down to meet it, and the worst case at the new crossing
is over the bound. There is nowhere better to put the line.

## Zero, NaN and infinity

Zero is the one sample no level describes, so it takes index zero and comes back
exact.

A NaN has no magnitude to place. It lands on the same index and comes back as
zero, so a tile that means its NaNs puts the nodata codec in front. The sign of a
negative zero is not kept.

An infinity is pinned to the top level rather than run through a transform on an
exponent field of all ones. Case 3 gives back the largest finite value of the
type, cases 1 and 2 the largest whole number it carries. `quant_log_scan` leaves
infinities out of the range it reports, so neither `STORE=VALUES` refusal below
ever sees one.

## What it refuses

A bound under what the type can rebuild.

    LOG:MAX_ERROR=0.001% on float16

    a MAX_ERROR of 0.001% is at or below what this type rebuilds to, which
    bottoms out near 0.0488281%

A bound needing more index levels than the stream width holds. The whole type has
to fit, since the grid never reads the tile and so cannot be trimmed to the range
that happens to be present.

    LOG:MAX_ERROR=0.00001% on float32

    a MAX_ERROR of 1e-05% needs more levels than a 4-byte index carries

`STORE=VALUES` on a tile that reaches below the crossing, where rounding a float
to a whole number already costs more than the bound.

    LOG:MAX_ERROR=1%,STORE=VALUES on a DEM in metres

    STORE=VALUES rounds to whole numbers, which holds a MAX_ERROR of 1% only
    at or above 100.249, and this tile reaches 0.4

`STORE=VALUES` where the reconstruction would run past the largest whole number
the float type carries, which is `2^11`, `2^24` or `2^53`.

    STORE=VALUES needs a reconstruction exact in the output type, and
    3.02202e+07 is past the 1.67772e+07 it carries

None of these quietly becomes case 3. Whether `STORE=VALUES` fits depends on the
tile, so falling back would make the grid depend on the tile, and the next
section is the whole reason that is not allowed.

## Nothing about the grid comes from the tile

Not the anchor, not the step. The same value rebuilds the same way whatever
raster it was cut from and at whatever chunk size, so a scene compressed at one
tile size and again at another gives the same numbers.

The encoder does read the tile, for three things, and none of them moves a level.
Whether any sample is negative, which sets the floor the decoder clamps against.
The smallest and the largest magnitude, which decide the two `STORE=VALUES`
refusals above. That is all of it.

## The header

Ten bytes, little endian.

    byte 0     element type. 0 to 3 are u8, u16, u32, u64, 4 to 7 the signed
               widths in the same order, then 8 f16, 9 f32, 10 f64
    byte 1     flags. bit 0, the encoder saw no negative sample, so the decoder
               floors at zero. bit 1, the stream holds the reconstruction
    bytes 2-9  step, an IEEE double

A frame is corrupt when the type byte falls outside 0 to 10, when a flag bit
above bit 1 is set, when bit 1 is clear on an integer type, or when the step is
not finite or is at or below zero. Both kernels check all four themselves rather
than trusting a binding to have done it. Finite does not follow from positive. An
infinite step leaves the quotient inside exp2 a NaN, and the conversion after it
is undefined.

## Where the bound ends

At the smallest normal of a float type. A subnormal `k * 2^E` has neighbours a
flat `2^E` apart, so rounding it costs `0.5/k`, and that grows without limit as k
falls. No grid reaches down there and no constant covers it. `quant_log_scan`
reports a tile that goes below, so the caller can decide what to do about it.

Everything at or above the smallest normal is covered, checked one value at a
time over the whole of uint8, uint16, int16, float16 and the normal float32.

    LOG:MAX_ERROR=5%     worst 4.999998e-02   over the bound 0
    LOG:MAX_ERROR=1%     worst 9.999988e-03   over the bound 0
    LOG:MAX_ERROR=0.1%   worst 9.999960e-04   over the bound 0

4261412864 normal float32 values each time.

## Why integers are a first-class case

Most lossy compressors for scientific data will not take an integer with a
pointwise relative bound. SZ2 prints a message and exits on one, and SZ3 dropped
the mode altogether, so its `REL` is a flat bound scaled by the range of the
data. Earth observation is full of uint16 rasters, so case 1 is the one that
matters here rather than an afterthought.

## Both transcendentals are written out

A frame is written on one machine and read on another, and libm is not correctly
rounded, so two platforms can disagree by an ulp and rebuild the same index two
ways. `log2` and `exp2` live in `quant_log_math.h` in plain IEEE arithmetic, and
with `-ffp-contract=off` they evaluate the same anywhere.

Neither kernel calls libm at all. Only the resolver does, once per tile, and what
it produces is the `step` in the header, which both sides then read. `log1p` is
not correctly rounded either, so two encoders on two libm can settle on steps an
ulp apart for the same recipe. That is why the step travels in the header instead
of being worked out again on the way back.

The forward transform takes sixty-four points across the mantissa, each with its
log2 and its reciprocal, so the mantissa costs a multiply and a short series
instead of a call. Measured against libm across the whole range of each type.

    log2  f16   worst 7.1e-15   budget 3.6e-14
    log2  f32   worst 5.7e-14   budget 2.5e-13
    log2  f64   worst 2.3e-13   budget 1.9e-12
    exp2        worst 2.2e-16   budget 1.3e-12

## Speed

The decoder for case 3 builds the levels the tile actually uses, once, at the
output width and already folded onto the output range. The table covers index
zero and both signs, so the loop has no branch in it. Above 64 kB of table the
long way is taken instead.

The encoder for cases 1 and 2 caches the value grid the same way, and above a
megabyte of table it builds each level on the spot. A tight bound on a wide type
wants millions of levels, which is a size to decline to allocate and not a reason
to decline the recipe. Which path is taken reaches no level.

    8M elements, one per cent, on one core

    case 1  uint16   encode   367 MB/s   decode  12648 MB/s
    case 2  float64  encode  1278 MB/s   decode   5820 MB/s
    case 3  float32  encode   649 MB/s   decode   2777 MB/s
