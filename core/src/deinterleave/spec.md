## Deinterleave Decoder Specification

The `deinterleave` codec is lossless, in the `geozl.lossless` namespace, CTid
`0x72D704`. It splits one lane interleaved numeric stream into two by position on
encode, and interleaves the two back into one on decode. It is the inverse role
of OpenZL's built in interleave, which combines on encode and splits on decode.

The codec is dtype agnostic. It only moves elements, it never reads their value,
so a caller carries any lane semantics itself. The common use is complex data,
where the two lanes are the real and the imaginary component, kept apart so a
spatial predictor downstream never crosses the component boundary. The complex
type itself is not the codec's concern, the consumer records it.

### Inputs
The decoder takes two numeric streams of the same element width, the even lane
and the odd lane. If the total output has N elements, the even lane has
`ceil(N/2)` elements and the odd lane has `floor(N/2)`. It is an error for the
two lanes to differ in element width, or in count by more than one.

### Codec Header
None. The decoder reconstructs the element order from the two lane counts alone.

### Decoding
The output is a single numeric stream of `N` elements. Element `2k` is taken from
the even lane at index `k`, element `2k+1` from the odd lane at index `k`.

Consider an even lane {a0, a1, a2} and an odd lane {b0, b1}. The decoded stream is
{a0, b0, a1, b1, a2}.

### Outputs
A single numeric stream of the same element width as the two inputs, with a count
equal to the sum of the two lane counts.
