import os
import struct
import subprocess
import sys

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_DISABLE = 2  # ZL_TernaryParam_disable

# quant_linear writes -step on integers, where the reconstruction travels in
# place of the index. Derived, not written out, so a change to that rule breaks
# the anchor loudly instead of quietly stopping these cases from running.
_QL_MAX_ERROR = 50
_QL_SCALE = geozl.lossy.QuantLinear(_QL_MAX_ERROR, np.uint16)._scale


def _compress(node, arr):
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = node(c, g)
    c.select_starting_graph(g)
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


def _place(build, needle, tries=64):
    """A frame where needle appears exactly once, plus where it landed.

    Absent means the header no longer holds what this test thinks it does,
    which is a bug worth failing on rather than skipping. Two copies is only
    the payload colliding, and another tile shifts it.
    """
    for i in range(tries):
        frame = bytearray(build(i))
        at = frame.find(needle)
        if at < 0:
            sys.exit(4)
        if frame.find(needle, at + 1) < 0:
            return frame, at
    sys.exit(2)


# width that does not divide the tile, the encoder must reject it
def _case_width_encode():
    arr = np.arange(200, dtype=np.uint16)
    _compress(geozl.lossless.DeltaW(7), arr)


# same width fault, forged into the codec header
def _case_width_decode():
    def build(i):
        return _compress(geozl.lossless.DeltaW(200),
                         np.arange(200, dtype=np.uint16) + i)

    frame, at = _place(build, struct.pack("<I", 200))
    frame[at:at + 4] = struct.pack("<I", 7)
    _decode(bytes(frame))


def _ql_build(i):
    arr = np.arange(256, dtype=np.uint16).reshape(16, 16) + i
    return _compress(geozl.lossy.QuantLinear(_QL_MAX_ERROR, np.uint16), arr)


# dtype byte outside the type enum. The header is <Bd, so it sits right before
# the scale.
def _case_dtype():
    frame, at = _place(_ql_build, struct.pack("<d", _QL_SCALE))
    frame[at - 1] = 200
    _decode(bytes(frame))


# non-finite scale the integer path cannot truncate to int64
def _case_scale():
    frame, at = _place(_ql_build, struct.pack("<d", _QL_SCALE))
    frame[at:at + 8] = struct.pack("<d", float("inf"))
    _decode(bytes(frame))


# shift wider than the accumulator. Two rows force the planar default, so the
# coeffs are a known anchor with the shift byte right before them.
def _case_shift():
    def build(i):
        arr = np.tile(np.arange(1, 101, dtype=np.uint16), (2, 1)) + i
        return _compress(geozl.lossless.WpStatic(100), arr)

    frame, at = _place(build, struct.pack("<4h", 1, -1, 0, 0))
    frame[at - 1] = 200
    _decode(bytes(frame))


_CASES = {
    "width_encode": _case_width_encode,
    "width_decode": _case_width_decode,
    "dtype": _case_dtype,
    "scale": _case_scale,
    "shift": _case_shift,
}


# Each case runs in its own process so one bad frame cannot take down the run.
# macOS strips the dyld preload from the child, so reinject the runtime path.
def _child(name):
    env = dict(os.environ)
    rt = env.get("GEOZL_ASAN_RT")
    if rt:
        var = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
        env[var] = rt
    return subprocess.run(
        [sys.executable, __file__, name], capture_output=True, text=True, env=env)


@pytest.mark.parametrize("name", list(_CASES))
def test_decode_rejects_bad_header(name):
    r = _child(name)
    assert r.returncode != 4, \
        "the anchor is gone, this case has stopped testing anything"
    if r.returncode == 2:
        pytest.skip("could not place the corruption")
    assert r.returncode == 0, r.stderr or f"exit {r.returncode}"


if __name__ == "__main__":
    # 0 rejected cleanly, 3 slipped through, 4 the anchor is gone, a signal
    # means it corrupted memory
    try:
        _CASES[sys.argv[1]]()
    except SystemExit:
        raise
    except BaseException:
        sys.exit(0)
    sys.exit(3)