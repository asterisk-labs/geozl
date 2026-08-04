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


def test_the_dtype_width_table_matches_the_one_in_c():
    """A quant header carries a dtype code and the decoder sizes its output
    against this table. The same one is written out in quant_linear_dtype.h,
    quant_log_dtype.h and quant_sqrt_dtype.h, none of them reachable from here,
    so the literal is what anchors the four copies together."""
    from geozl._dtype import dtype_width
    assert [dtype_width(c) for c in range(11)] == [1, 2, 4, 8, 1, 2, 4, 8,
                                                   2, 4, 8]
    assert dtype_width(11) is None
    assert dtype_width(255) is None


def _codec_header(decoder_cls, frame):
    """The header bytes the encoder wrote, read back through its own decoder.
    A step copied from the resolver here would rot into a test that forges the
    payload instead."""
    seen = []

    class Spy(decoder_cls):
        def decode(self, state):
            seen.append(bytes(state.codec_header))
            super().decode(state)

    d = zl.DCtx()
    d.set_parameter(zl.DParam.CheckCompressedChecksum, _DISABLE)
    d.set_parameter(zl.DParam.CheckContentChecksum, _DISABLE)
    d.register_custom_decoder(Spy())
    d.decompress(frame)
    return seen[0]


# uint16 keeps every curve off the raster, so one recipe writes one header.
_QUANTIZERS = [
    ("quant_linear",
     lambda: geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.uint16),
     geozl.lossy.QuantLinearDecoder),
    ("quant_log",
     lambda: geozl.lossy.QuantLog("LOG:MAX_ERROR=5%", np.uint16),
     geozl.lossy.QuantLogDecoder),
    ("quant_sqrt",
     lambda: geozl.lossy.QuantSqrt("SQRT:MAX_ERROR=0.5N,A=100,B=1", np.uint16),
     geozl.lossy.QuantSqrtDecoder),
]
_QUANT_ARGS = [(m, d) for _, m, d in _QUANTIZERS]
_QUANT_IDS = [n for n, _, _ in _QUANTIZERS]


def _quant_build(make):
    def build(i):
        return _frame(make(), np.arange(256, dtype=np.uint16) + i)

    return build


@pytest.mark.parametrize("make,decoder", _QUANT_ARGS, ids=_QUANT_IDS)
def test_quantizer_decoder_rejects_a_dtype_of_another_width(make, decoder):
    # float64 over a two byte stream, the one fault that reaches past the end
    # of the output the decoder sized
    build = _quant_build(make)
    frame = _forge(build, _codec_header(decoder, build(0)), 0, bytes([10]))
    with pytest.raises(Exception, match="bad dtype"):
        _decode(frame)


@pytest.mark.parametrize("make,decoder", _QUANT_ARGS, ids=_QUANT_IDS)
def test_quantizer_decoder_rejects_an_unknown_flag(make, decoder):
    build = _quant_build(make)
    header = _codec_header(decoder, build(0))
    frame = _forge(build, header, 1, bytes([header[1] | 0x80]))
    with pytest.raises(Exception, match="refused this codec header"):
        _decode(frame)


@pytest.mark.parametrize("make,decoder", _QUANT_ARGS, ids=_QUANT_IDS)
def test_quantizer_decoder_rejects_a_step_of_zero(make, decoder):
    # every index folds onto one level unless the kernel refuses
    build = _quant_build(make)
    frame = _forge(build, _codec_header(decoder, build(0)), 2,
                   struct.pack("<d", 0.0))
    with pytest.raises(Exception, match="refused this codec header"):
        _decode(frame)


def test_quant_sqrt_decoder_rejects_a_negative_offset():
    # the curve anchor, which the other two headers do not carry
    make, decoder = _QUANT_ARGS[2]
    build = _quant_build(make)
    frame = _forge(build, _codec_header(decoder, build(0)), 10,
                   struct.pack("<d", -1.0))
    with pytest.raises(Exception, match="refused this codec header"):
        _decode(frame)
