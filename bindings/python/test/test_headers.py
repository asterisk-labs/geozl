import struct

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_DISABLE = 2  # ZL_TernaryParam_disable

# quant wire codes, uint8..uint64 then int8..int64, floats after
_FLOAT32_CODE = 9


def _frame(node, arr):
    c = zl.Compressor()
    c.select_starting_graph(node(c, zl.graphs.Compress()(c)))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    cc.set_parameter(zl.CParam.ContentChecksum, _DISABLE)
    cc.set_parameter(zl.CParam.CompressedChecksum, _DISABLE)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def _decode(frame):
    d = zl.DCtx()
    d.set_parameter(zl.DParam.CheckCompressedChecksum, _DISABLE)
    d.set_parameter(zl.DParam.CheckContentChecksum, _DISABLE)
    geozl.register_decoders(d)
    return d.decompress(frame)


def _forge(build, anchor, offset, replacement, tries=64):
    """Rewrite part of a codec header in a real frame.

    anchor is a byte pattern from the header that has to appear exactly once,
    since corrupting the wrong copy would test nothing. offset is where the
    replacement goes relative to it. The payload moves with the tile, so a
    collision on one tile is gone on the next and build(i) just varies it.
    """
    for i in range(tries):
        frame = bytearray(build(i))
        at = frame.find(anchor)
        if at < 0 or frame.find(anchor, at + 1) >= 0:
            continue
        start = at + offset
        frame[start:start + len(replacement)] = replacement
        return bytes(frame)
    pytest.fail(f"no unique anchor for {anchor!r} in {tries} tiles")


def test_predictor_rejects_a_width_that_does_not_tile():
    # 200 samples cannot be laid out 7 wide
    with pytest.raises(Exception, match="does not tile"):
        _frame(geozl.lossless.Planar(7), np.arange(200, dtype=np.uint16))


def test_predictor_decoder_rejects_a_forged_row_width():
    # the header is one uint32 row width, and 7 does not tile 200, so the
    # decoder has to refuse what the encoder would never have written
    def build(i):
        return _frame(geozl.lossless.Planar(200),
                      np.arange(200, dtype=np.uint16) + i)

    frame = _forge(build, struct.pack("<I", 200), 0, struct.pack("<I", 7))
    with pytest.raises(Exception, match="bad row width"):
        _decode(frame)


def test_wp_static_rejects_a_width_that_does_not_tile():
    with pytest.raises(Exception, match="does not tile"):
        _frame(geozl.lossless.WpStatic(7), np.arange(200, dtype=np.uint16))


def _wp_build(i):
    # two rows keep the trainer on its planar default, so the coefficients stay
    # a known anchor and the rest of the header sits at a fixed offset from them
    arr = np.tile(np.arange(1, 101, dtype=np.uint16), (2, 1)) + i
    return _frame(geozl.lossless.WpStatic(100), arr)


# the header is <IB4h: row width, shift, then the four coefficients
_WP_COEFFS = struct.pack("<4h", 1, -1, 0, 0)


def test_wp_static_decoder_rejects_a_forged_row_width():
    frame = _forge(_wp_build, _WP_COEFFS, -5, struct.pack("<I", 7))
    with pytest.raises(Exception, match="bad row width"):
        _decode(frame)


def test_wp_static_decoder_rejects_a_shift_past_the_accumulator():
    frame = _forge(_wp_build, _WP_COEFFS, -1, bytes([200]))
    with pytest.raises(Exception, match="shift"):
        _decode(frame)


def test_quant_rejects_a_dtype_that_is_not_the_sample_width():
    # uint16 says two bytes per sample, the tile gives one
    with pytest.raises(Exception, match="does not match"):
        _frame(geozl.lossy.Quant("abs:4", np.arange(64, dtype=np.uint16)),
               np.arange(64, dtype=np.uint8))


def _q_build(i):
    arr = np.arange(256, dtype=np.uint16).reshape(16, 16) + i
    return _frame(geozl.lossy.Quant("abs:50", arr), arr)


# the header is <BBBddQ: dtype, curve and flags, then step and offset as
# doubles and nsub as a uint64. An error of 50 makes the step 100, and the step
# is the only field with a value worth anchoring on, so dtype, curve and flags
# sit three, two and one byte before it.
_Q_STEP = struct.pack("<d", 100.0)


def test_quant_decoder_rejects_a_dtype_outside_the_enum():
    frame = _forge(_q_build, _Q_STEP, -3, bytes([200]))
    with pytest.raises(Exception, match="bad dtype"):
        _decode(frame)


def test_quant_decoder_rejects_a_curve_it_does_not_know():
    frame = _forge(_q_build, _Q_STEP, -2, bytes([200]))
    with pytest.raises(Exception, match="bad curve"):
        _decode(frame)


def test_quant_decoder_rejects_a_non_finite_step():
    frame = _forge(_q_build, _Q_STEP, 0, struct.pack("<d", float("inf")))
    with pytest.raises(Exception, match="bad step"):
        _decode(frame)


def test_quant_decoder_rejects_a_stored_reconstruction_on_a_float():
    # the flag says the stream already holds the reconstruction, which only an
    # integer stream can carry. float32 is four bytes like uint32, so the width
    # check passes and this guard is the one that has to fire
    def build(i):
        arr = np.arange(256, dtype=np.uint32).reshape(16, 16) + i
        return _frame(geozl.lossy.Quant("abs:50", arr), arr)

    frame = _forge(build, _Q_STEP, -3, bytes([_FLOAT32_CODE]))
    with pytest.raises(Exception, match="integer type"):
        _decode(frame)


def test_quant_rejects_a_dtype_it_has_no_kernel_for():
    with pytest.raises(ValueError, match="does not support dtype"):
        geozl.lossy.Quant("abs:4", np.zeros((8, 8), np.bool_))


def test_quant_rejects_a_non_positive_error():
    with pytest.raises(ValueError, match="must be positive"):
        geozl.lossy.Quant("abs:0", np.arange(64, dtype=np.uint16))


def test_quant_rejects_a_bare_number():
    with pytest.raises(ValueError, match="recipe string"):
        geozl.lossy.Quant(4, np.arange(64, dtype=np.uint16))


def test_quant_rejects_a_relative_bound_without_the_percent_sign():
    # "rel:1" reads as a bound of one, a hundred percent, which is a plausible
    # typo for one percent and would quantize far coarser than intended
    with pytest.raises(ValueError, match="percentage"):
        geozl.lossy.Quant("rel:1", np.arange(64, dtype=np.uint16))