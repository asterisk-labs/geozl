from ._predictor import predictor

# header cap is 13 bytes, a uint32 width, a uint8 shift and four int16
# coefficients. The layout lives in C, wp_static fits its own kernel there.
WpStaticInt, WpStaticDecoder = predictor(
    0x72D707, "geozl.lossless.wp_static",
    "geozl_wp_static_encode_auto", "geozl_wp_static_decode_auto", 13)
