import importlib

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_ffi = importlib.import_module("geozl._ffi")
_pfor = importlib.import_module("geozl.lossless.pfor")
_fused = importlib.import_module("geozl.lossless.planar_zigzag_pfor")
_ptr, lib = _ffi._ptr, _ffi.lib


def _frame(array, width, planes=1):
    compressor = zl.Compressor()
    graph = geozl.lossless.PlanarZigzagPfor(width, planes=planes)(
        compressor, zl.graphs.Store()(compressor)
    )
    compressor.select_starting_graph(graph)
    context = zl.CCtx()
    context.ref_compressor(compressor)
    context.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(array).reshape(-1)
    return bytes(context.compress([zl.Input(zl.Type.Numeric, flat)]))


@pytest.mark.parametrize(
    "dtype", ["uint8", "int16", "uint32", "uint64", "float32", "float64"]
)
@pytest.mark.parametrize("shape", [(13, 17), (2, 257), (1024, 1)])
def test_python_codec_round_trips_through_both_decoder_registries(dtype, shape):
    values = np.arange(np.prod(shape), dtype=np.uint64)
    array = (values * values + 17).astype(dtype).reshape(shape)
    frame = _frame(array, shape[-1])

    python_dctx = zl.DCtx()
    geozl.register_decoders(python_dctx)
    python_out = python_dctx.decompress(frame)[0].content.as_nparray()
    assert python_out.view(array.dtype).tobytes() == array.tobytes()

    c_out = geozl.decompress(frame)
    assert c_out.view(array.dtype).tobytes() == array.tobytes()


def test_stacked_planes_reset_inside_a_pfor_block():
    array = np.arange(3 * 13 * 17, dtype=np.int16).reshape(3, 13, 17)
    frame = _frame(array, 17, planes=3)
    out = geozl.decompress(frame).view(array.dtype).reshape(array.shape)
    assert np.array_equal(out, array)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
def test_raw_payload_equals_planar_zigzag_then_pfor(dtype):
    array = (np.arange(3 * 13 * 17, dtype=np.uint64) ** 2).astype(dtype)
    array = array.reshape(3, 13, 17)
    flat = np.ascontiguousarray(array).reshape(-1)
    transformed = np.empty_like(flat)
    per_plane = flat.size // 3
    for plane in range(3):
        start = plane * per_plane
        stop = start + per_plane
        assert lib.planar_zigzag_encode(
            _ptr(transformed[start:stop]),
            _ptr(flat[start:stop]),
            17,
            per_plane,
            flat.dtype.itemsize,
        ) == 0

    stable = _pfor.encode(transformed)
    fused = _fused.encode(array, 17, planes=3)
    assert fused == stable


def test_raw_decoder_rejects_truncation_and_trailing_bytes():
    array = np.arange(2 * 13 * 17, dtype=np.uint16).reshape(2, 13, 17)
    payload = _fused.encode(array, 17, planes=2)
    with pytest.raises(ValueError, match="corrupt"):
        _fused.decode(payload[:-1], array.size, array.dtype, 17, planes=2)
    with pytest.raises(ValueError, match="corrupt"):
        _fused.decode(payload + b"\x00", array.size, array.dtype, 17, planes=2)
