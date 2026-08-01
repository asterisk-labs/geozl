## Quant-Linear Decoder Specification

A uniform grid anchored at zero, for a flat absolute bound.

    LINEAR:MAX_ERROR=V                 |x - x^| <= V
    LINEAR:MAX_ERROR=V,STORE=VALUES    the stream carries the reconstruction

### Inputs
One numeric stream of `nbElts` elements, at the element width of the original
type.

### Codec Header
Little endian, 10 bytes.

- byte 0, the original element type. 0 to 3 are u8, u16, u32, u64, 4 to 7 the
  signed widths in the same order, then 8 f16, 9 f32, 10 f64.
- byte 1, flags. Bit 0, the encoder found no negative sample, so the decoder
  floors at zero rather than at the type minimum. Bit 1, the stream carries the
  reconstruction rather than the index.
- bytes 2 to 9, `step`, an IEEE double.

`inf` and `nan` in `step` are corrupt, so is a `step` at or below zero, a flag bit
above bit 1, bit 1 clear on an integer type, and bit 1 set on a float type with a
fractional `step`.

### The three cases
The stream is always integer. What it holds depends on whether the reconstruction
is a whole number, which is the whole reason there are two float paths.

| input | recipe | stream holds | decode | step | rounds |
| --- | --- | --- | --- | --- | --- |
| integer | any | reconstruction | `x = q` | `max(1, floor(2V))` | no |
| float | `STORE=VALUES` | reconstruction | `x = (float)q` | `floor(2V)` | no |
| float | default | index | `x = q * step` | `2 * (V - eps * maxAbs)` | yes |

The result is clamped into the output range, floored at zero instead when bit 0 is
set, and rounded once to the output width. Clamping only shortens `|x - x^|`, since
the interval it clamps into holds `x`, so the bound survives it.

**Integer.** Sentinel-2 DN, `MAX_ERROR=5`, `step = 10`. A multiple of 10 is an
integer, so it rides the stream and the decoder copies. Free, since scaling every
symbol by a constant leaves the entropy of the residual downstream unchanged.

    in       1     83    166    248
    stream   0     80    170    250
    out      0     80    170    250

**Float whose grid is whole numbers.** A DEM in metres that arrived as float32,
`MAX_ERROR=0.5,STORE=VALUES`, `step = 1`. Every value goes to the nearest metre, in
either direction, and the error is the fraction it gave up.

    in     -420.3      0.4   1250.2   3565.9   3566.4
    stream   -420        0     1250     3566     3566
    out    -420.0      0.0   1250.0   3566.0   3566.0
    error     0.3      0.4      0.2      0.1      0.4

The reconstruction is a whole number, so it rides the stream and the decoder casts
rather than multiplies. That cast is exact, so nothing rounds after the grid. Data
already on whole metres comes back untouched.

**Float not holding whole numbers.** Reflectance, `MAX_ERROR=0.0005`,
`step = 0.00099997`. Here `q * step` is fractional, so the stream carries the index
and the decoder multiplies. `STORE=VALUES` is refused, since `floor(2 * 0.0005)` is
zero and there is no whole step to put a grid on.

    in     0.000000   0.083400   0.165700   0.247900
    stream        0         83        166        248
    out    0.000000   0.082998   0.165995   0.247993

### Where rounding bites
Only the index path. `(float)(q * step)` gives up `eps * |x^|`, so the error at
index `q` is `step/2 + eps * |q * step|`. The second term grows with `q`, so the
step gives up the worst of it, which puts `maxAbs` in the formula. Without it a
float32 near a million under `MAX_ERROR=0.05` misses by a quarter, because float32
there only resolves 0.0625.

    LINEAR:MAX_ERROR=0.05 on float32 reaching 1e6
      a MAX_ERROR of 0.05 is at or below the rounding of the output type at 1e+06

That `maxAbs` is also the one thing here that reads the tile. Two tiles with
different maxima do not share a grid, so the same value can reconstruct two ways,
up to twice the bound apart, which shows up when a raster is compressed at one
chunk size and again at another and when one outlying sample lifts the maximum.
The other two cases have no `maxAbs` and no rounding after the grid, so they are
the same grid however the raster is cut.

`STORE=VALUES` has its own limit. A float holds every integer only up to `2^11`,
`2^24`, `2^53` for f16, f32, f64, past which the cast back would round.

    LINEAR:MAX_ERROR=0.5,STORE=VALUES on float32 reaching 3e7
      STORE=VALUES needs a reconstruction exact in the output type,
      and 3e+07 is past the 1.67772e+07 it carries

Both are refusals, never a quiet fallback to the other path. Whether
`STORE=VALUES` fits depends on the range, so falling back would put the grid's
dependence on the tile straight back in.

### Picking a whole step

With a step `s` the error is at most `s/2`, so the step a bound of `V` wants is
`2V`. A grid of whole units rarely gets it and has to take the neighbour, and it
has to take the one below.

`MAX_ERROR=0.94` wants `1.88`, so the choice is 1 or 2:

    step 2      in    0  1  2  3  4  5  6
                out   0  2  2  4  4  6  6      worst 1.0, over the 0.94 declared

    step 1      in    0  1  2  3  4  5  6
                out   0  1  2  3  4  5  6      worst 0, under it

Rounding up breaks the bound. Truncating only gives up ratio, so the step is
`floor(2V)`, never `nearbyint(2V)`.

| `MAX_ERROR` | `2V` | integer | float, `STORE=VALUES` |
| --- | --- | --- | --- |
| 0.4 | 0.8 | step 1, error 0 | refused |
| 0.5 | 1 | step 1, error 0 | step 1, error <= 0.5 |
| 0.94 | 1.88 | step 1, error 0 | step 1, error <= 0.5 |
| 1 | 2 | step 2, error <= 1 | step 2, error <= 1 |
| 12.75 | 25.5 | step 25, error <= 12.5 | step 25, error <= 12.5 |

The two columns part at the bottom because a step of one means different things.
On an integer the data is already on the grid, `q = round(x/1) = x`, so nothing is
lost and any bound holds:

    in    0  1  2  3  4  5  6
    out   0  1  2  3  4  5  6

On a float it is not, and a step of one costs up to 0.5:

    in    0.0  0.3  0.7  1.2  1.5  2.4  2.9
    out   0.0  0.0  1.0  1.0  2.0  2.0  3.0

So one is a safe floor on an integer and a refusal on a float, where a
`MAX_ERROR` under 0.5 has no whole step that holds it.