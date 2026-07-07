import struct
import subprocess
import sys

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_DISABLE = 2  # ZL_TernaryParam_disable


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


def _locate(frame, needle):
    at = frame.find(needle)
    if at < 0 or frame.find(needle, at + 1) >= 0:
        return -1
    return at


# A width that does not divide the element count. The encoder must reject it
# rather than run the kernel off the end of the row.
def _case_width_encode():
    arr = np.arange(200, dtype=np.uint16)
    _compress(geozl.lossless.DeltaW(7), arr)


# Same width fault reached through the codec header a decoder trusts.
def _case_width_decode():
    arr = np.arange(200, dtype=np.uint16)
    frame = bytearray(_compress(geozl.lossless.DeltaW(200), arr))
    at = _locate(frame, struct.pack("<I", 200))
    if at < 0:
        sys.exit(2)
    frame[at:at + 4] = struct.pack("<I", 7)
    _decode(bytes(frame))


# A dtype byte outside the type enum, which the passthrough path uses to size a
# copy and to pick the element width.
def _case_dtype():
    arr = np.arange(256, dtype=np.uint16).reshape(16, 16)
    frame = bytearray(_compress(geozl.lossy.QuantLinear(50, np.uint16), arr))
    at = _locate(frame, struct.pack("<d", 100.0))
    if at <= 0:
        sys.exit(2)
    frame[at - 1] = 200
    _decode(bytes(frame))


# A shift wider than the accumulator, an undefined shift count in the predictor.
def _case_shift():
    arr = np.tile(np.arange(101, dtype=np.uint16), (7, 1))
    frame = bytearray(_compress(geozl.lossless.WpStatic(101), arr))
    at = _locate(frame, struct.pack("<I", 101))
    if at < 0:
        sys.exit(2)
    frame[at + 4] = 200
    _decode(bytes(frame))


_CASES = {
    "width_encode": _case_width_encode,
    "width_decode": _case_width_decode,
    "dtype": _case_dtype,
    "shift": _case_shift,
}


# The child exits 0 when the fault is rejected cleanly, 3 when it slips through,
# and dies on a signal when it corrupts memory. Isolating it keeps one bad frame
# from taking down the whole run.
def _child(name):
    return subprocess.run(
        [sys.executable, __file__, name], capture_output=True, text=True)


@pytest.mark.parametrize("name", list(_CASES))
def test_decode_rejects_bad_header(name):
    r = _child(name)
    if r.returncode == 2:
        pytest.skip("could not place the corruption")
    assert r.returncode == 0, r.stderr or f"exit {r.returncode}"


if __name__ == "__main__":
    try:
        _CASES[sys.argv[1]]()
    except SystemExit:
        raise
    except BaseException:
        sys.exit(0)
    sys.exit(3)