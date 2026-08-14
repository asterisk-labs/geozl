## planar Decoder Specification
### Inputs
A single numeric stream of 8, 16, 32 or 64-bit integers holding the planar residual plane in row major order.

### Codec Header
A single uint32, little endian, the row width in samples. The number of rows is the element count divided by the width.

The header is four bytes for one plane and eight for more, the extra uint32 being the plane count, so a four byte header still means one plane. Each plane is predicted on its own, its first row taking N and NW as zero rather than reaching into the plane before it. A count that does not split the elements into whole-row planes is corruption.

### Decoding
The predictor is W + N - NW, where W is the left reconstructed sample, N the sample above and NW the sample above left. Edge neighbors are zero, so the first row of each plane reduces to the horizontal predictor and column zero to the vertical one. Each row is reconstructed by adding N - NW from the row above into the residual, then a prefix sum over the row resolves the W chain, all in native width modular arithmetic.

### Outputs
A single numeric stream of the same element width and the same length as the input.
