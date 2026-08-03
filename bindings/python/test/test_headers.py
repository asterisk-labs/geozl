import struct

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_DISABLE = 2  # ZL_TernaryParam_disable

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


# The quant header tests went with geozl.lossy.Quant. The three codecs that
# replaced it carry their header checks in C, over their decode bindings and in
# their fuzzers.
