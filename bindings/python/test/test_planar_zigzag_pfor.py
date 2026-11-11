import importlib

import numpy as np
import pytest

zl = pytest.importorskip("openzl.ext")
geozl = pytest.importorskip("geozl")

_ffi = importlib.import_module("geozl._ffi")
_ptr, lib = _ffi._ptr, _ffi.lib

try:
    _ffi._load_lib_full()
    _HAVE_FULL = True
except OSError:  # pragma: no cover - depends on how the build was configured
    _HAVE_FULL = False

needs_full = pytest.mark.skipif(
    not _HAVE_FULL, reason="libgeozl not built, rebuild with FULL=ON"
)


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


def _python_decompress(frame):
    dctx = zl.DCtx()
    geozl.register_decoders(dctx)
    return dctx.decompress(frame)[0].content.as_nparray()


_DECODERS = [
    pytest.param(_python_decompress, id="python"),
    pytest.param(geozl.decompress, marks=needs_full, id="c"),
]


@pytest.mark.parametrize("decoder", _DECODERS)
@pytest.mark.parametrize(
    "dtype", ["uint8", "int16", "uint32", "uint64", "float32", "float64"]
)
@pytest.mark.parametrize("shape", [(13, 17), (2, 257), (1024, 1), (3, 2100)])
def test_python_codec_round_trips_through_decoder_registry(decoder, dtype, shape):
    values = np.arange(np.prod(shape), dtype=np.uint64)
    array = (values * values + 17).astype(dtype).reshape(shape)
    frame = _frame(array, shape[-1])

    out = decoder(frame)
    assert out.view(array.dtype).tobytes() == array.tobytes()


@pytest.mark.parametrize("decoder", _DECODERS)
def test_stacked_planes_predict_apart(decoder):
    array = np.arange(3 * 13 * 17, dtype=np.int16).reshape(3, 13, 17)
    frame = _frame(array, 17, planes=3)
    out = decoder(frame).view(array.dtype).reshape(array.shape)
    assert np.array_equal(out, array)


def _band_heights(rows):
    heights = [16] * (rows // 16)
    for h in (8, 4, 2, 1):
        if rows % 16 & h:
            heights.append(h)
    return heights


STRIP = 1024


def _reference_payload(array, width, planes):
    """Build the payload independently from the fused codec."""
    flat = np.ascontiguousarray(array).reshape(-1)
    unsigned = flat.view(f"u{flat.dtype.itemsize}")
    bits = 8 * flat.dtype.itemsize
    plane = unsigned.reshape(planes, -1, width)
    headers, bodies = [], []
    for p in range(planes):
        y0 = 0
        for h in _band_heights(plane.shape[1]):
            band = plane[p, y0:y0 + h].astype(np.uint64)
            west = np.zeros_like(band)
            west[:, 1:] = band[:, :-1]
            north = np.zeros_like(band)
            north[1:] = band[:-1]
            northwest = np.zeros_like(band)
            northwest[1:, 1:] = band[:-1, :-1]
            # Prediction restarts at each strip.
            west[:, ::STRIP] = 0
            northwest[:, ::STRIP] = 0
            mask = np.uint64((1 << bits) - 1)
            residual = (band - (west + north - northwest)) & mask
            sign = np.where(residual >> np.uint64(bits - 1), mask, np.uint64(0))
            encoded = ((residual << np.uint64(1)) ^ sign) & mask
            block_width = 256 // h
            for x0 in range(0, width, block_width):
                slots = np.zeros(256, dtype=unsigned.dtype)
                for i in range(h):
                    for j in range(min(block_width, width - x0)):
                        slots[8 * (i + h * (j // 8)) + j % 8] = encoded[i, x0 + j]
                block = geozl.lossless.pfor_encode(slots)
                head = 3 if block[1] else 2
                headers.append(bytes([block[0], block[1], block[2] if block[1] else 0]))
                bodies.append(block[head:])
            y0 += h
    return b"".join(headers) + b"".join(bodies)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
@pytest.mark.parametrize("shape", [(3, 13, 17), (1, 20, 40), (2, 1, 300), (1, 33, 3),
                                   (1, 3, 2100)])
def test_raw_payload_follows_the_band_layout(dtype, shape):
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.uint64) ** 2) % np.uint64(2**61 - 1)
    array = values.astype(dtype).reshape(shape)
    planes, width = shape[0], shape[-1]
    fused = geozl.lossless.planar_zigzag_pfor_encode(array, width, planes=planes)
    assert fused == _reference_payload(array, width, planes)


def test_raw_decoder_rejects_truncation_and_trailing_bytes():
    array = np.arange(2 * 13 * 17, dtype=np.uint16).reshape(2, 13, 17)
    payload = geozl.lossless.planar_zigzag_pfor_encode(array, 17, planes=2)
    with pytest.raises(ValueError, match="corrupt"):
        geozl.lossless.planar_zigzag_pfor_decode(payload[:-1], array.size, array.dtype, 17,
                                                 planes=2)
    with pytest.raises(ValueError, match="corrupt"):
        geozl.lossless.planar_zigzag_pfor_decode(payload + b"\x00", array.size, array.dtype,
                                                 17, planes=2)


def test_raw_helpers_refuse_arrays_that_hold_references():
    values = np.array([1, None, 3, 4], dtype=object)
    with pytest.raises(ValueError, match="references, not samples"):
        geozl.lossless.pfor_encode(values)
    with pytest.raises(ValueError, match="references, not samples"):
        geozl.lossless.pfor_decode(b"", 4, object)
    with pytest.raises(ValueError, match="references, not samples"):
        geozl.lossless.planar_zigzag_pfor_encode(values, 2)
    with pytest.raises(ValueError, match="references, not samples"):
        geozl.lossless.planar_zigzag_pfor_decode(b"", 4, object, 2)
