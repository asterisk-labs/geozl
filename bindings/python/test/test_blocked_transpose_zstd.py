import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")


def _frame(arr, block_size):
    c = zl.Compressor()
    c.set_parameter(zl.CParam.CompressionLevel, 6)
    node = geozl.lossless.BlockedTransposeZstd(block_size)
    c.select_starting_graph(node(c, zl.graphs.Store()))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint32, np.float64])
@pytest.mark.parametrize("block_size", [127, 1024, 2 * 1024 * 1024])
def test_round_trip(dtype, block_size):
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 1000, 4099).astype(dtype)
    frame = _frame(arr, block_size)
    d = zl.DCtx()
    geozl.register_decoders(d)
    back = d.decompress(frame)[0].content.as_nparray().view(arr.dtype)
    assert np.array_equal(back, arr)


def test_c_reader_reads_python_writer():
    arr = np.arange(5000, dtype=np.uint16).reshape(50, 100)
    frame = _frame(arr, 1024)
    back = geozl.decompress(frame).view(arr.dtype).reshape(arr.shape)
    assert np.array_equal(back, arr)


def test_2d_c_writer_round_trips():
    arr = np.arange(5000, dtype=np.uint16).reshape(50, 100)
    frame = geozl.compress(
        arr, graph=geozl.graph(arr, method="id>blocked_transpose_zstd"))
    assert np.array_equal(
        geozl.decompress(frame).view(arr.dtype).reshape(arr.shape), arr)


def test_python_reader_reads_2d_c_writer():
    arr = np.arange(5000, dtype=np.uint16).reshape(50, 100)
    frame = geozl.compress(
        arr, graph=geozl.graph(arr, method="id>blocked_transpose_zstd"))
    d = zl.DCtx()
    geozl.register_decoders(d)
    back = d.decompress(frame)[0].content.as_nparray().view(arr.dtype)
    assert np.array_equal(back.reshape(arr.shape), arr)


def test_corrupt_payload_is_rejected():
    arr = np.arange(5000, dtype=np.uint16)
    frame = bytearray(_frame(arr, 1024))
    frame[-20] ^= 0x80
    d = zl.DCtx()
    geozl.register_decoders(d)
    with pytest.raises(RuntimeError, match="checksum|corrupt"):
        d.decompress(bytes(frame))


def test_bad_block_size_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        geozl.lossless.BlockedTransposeZstd(0)
