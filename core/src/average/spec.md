## average Decoder Specification
### Inputs
A single numeric stream of 8, 16, 32 or 64-bit integers holding the Average residual plane in row major order.

### Codec Header
A single uint32, little endian, the row width in samples. The number of rows is the element count divided by the width.

The header is four bytes for one plane and eight for more, the extra uint32 being the plane count, so a four byte header still means one plane. Each plane is predicted on its own, its first row taking N and NW as zero rather than reaching into the plane before it. A count that does not split the elements into whole-row planes is corruption.

### Decoding
The predictor is the floor average of the two neighbours, floor((W + N) / 2), where W is the left reconstructed sample and N the sample above. It is computed as (W >> 1) + (N >> 1) + (W & N & 1), which equals the floor average without overflowing the sample width. Edge neighbors are zero, so the first row of each plane predicts floor(W / 2) and the first column floor(N / 2). The prediction reads the reconstructed W, so each sample is reconstructed in order from its decoded neighbours, in native width modular arithmetic.

### Outputs
A single numeric stream of the same element width and the same length as the input.
