## med Decoder Specification
### Inputs
A single numeric stream of 8, 16, 32 or 64-bit integers holding the MED residual plane in row major order.

### Codec Header
A single uint32, little endian, the row width in samples. The number of rows is the element count divided by the width.

### Decoding
The predictor is the median edge detector of W, N, NW, where W is the left reconstructed sample, N the sample above and NW the sample above left. It is min(W,N) when NW is at least max(W,N), max(W,N) when NW is at most min(W,N), and the gradient W + N - NW otherwise. Edge neighbors are zero, so row zero reduces to the horizontal predictor and column zero to the vertical one. Because the prediction depends on W nonlinearly it is not a prefix sum, each sample is reconstructed in order from its decoded neighbours, in native width modular arithmetic.

### Outputs
A single numeric stream of the same element width and the same length as the input.
