## wp_static Decoder Specification

wp_static is a frozen adaptation of the JPEG XL self-correcting weighted predictor (ISO/IEC 18181-1, Annex E.3). The original blends sub-predictors with weights that adapt per sample from recent prediction errors. Freezing those weights removes the adaptation and leaves a fixed linear function of the causal neighbours, constrained so the left sample W enters with unit weight, which lets each row decode as a prefix sum as in planar.

### Inputs
A single numeric stream of 8, 16, 32 or 64-bit integers holding the residual plane in row major order.

### Codec Header
Little endian, in order: a uint32 row width in samples, a uint8 right shift, then four int16 coefficients cN, cNW, cNE, cNN. The number of rows is the element count divided by the width.

### Decoding
The prediction is W + K, with W the left reconstructed sample and K a linear kernel over the row above,

  K = (cN*N + cNW*NW + cNE*NE + cNN*NN + round) >> shift,

where N is the sample above, NW above left, NE above right, NN two above, round = shift ? 1 << (shift - 1) : 0, and any neighbour outside the plane is zero. The sum is accumulated in signed 32-bit for 8 and 16-bit samples and signed 64-bit for 32 and 64-bit, then normalized by an arithmetic right shift, so coefficients are small fixed point that keep the sum within that width. Zero edge neighbours make row zero the horizontal predictor and column zero the vertical part of the kernel. Each sample is reconstructed as res + W + K in native width modular arithmetic. K carries no left dependency, so a row folds K into the residual and resolves the W chain as a prefix sum, as in planar. The defaults cN=1, cNW=-1, cNE=0, cNN=0, shift=0 reproduce planar.

### Outputs
A single numeric stream of the same element width and the same length as the input.
