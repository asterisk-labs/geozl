"""planar_zigzag_pivco: Planar, Zigzag and PivCo Huffman per residual byte lane,
in one node. C only, since PivCo's kernels live inside OpenZL, so every frame
here is written and read by libgeozl."""

import numpy as np
import pytest

geozl = pytest.importorskip("geozl")

from geozl import _2d  # noqa: E402  after importorskip, on purpose

try:
    _2d._load_lib_full()
except OSError:  # pragma: no cover - depends on how the build was configured
    pytest.skip("libgeozl not built, rebuild with FULL=ON",
                allow_module_level=True)

METHOD = "planar>zigzag>pivco"


def _tile(shape=(48, 37), dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape[-2:])
    base = (x * 3 + y * 5) % 400 + rng.integers(0, 9, shape)
    return base.astype(dtype)


def _roundtrip(arr):
    frame = geozl.compress(arr, graph=geozl.graph(arr, METHOD))
    out = geozl.decompress(frame).view(arr.dtype).reshape(arr.shape)
    return frame, out


@pytest.mark.parametrize("dtype", [
    np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
    np.uint64, np.float32, np.float64,
])
@pytest.mark.parametrize("shape", [(48, 37), (1, 300), (300, 1), (3, 40, 29)])
def test_round_trips_every_width_and_shape(dtype, shape):
    arr = _tile(shape, dtype)
    _, out = _roundtrip(arr)
    assert np.array_equal(out, arr)


def test_a_constant_tile_needs_no_bitstream():
    arr = np.full((64, 64), 1234, dtype=np.uint16)
    frame, out = _roundtrip(arr)
    assert np.array_equal(out, arr)
    assert len(frame) < 200


def test_random_bytes_round_trip():
    arr = np.random.default_rng(3).integers(0, 2**16, (64, 64), dtype=np.uint16)
    _, out = _roundtrip(arr)
    assert np.array_equal(out, arr)


def test_a_tile_spanning_several_pivco_blocks_round_trips():
    # More than 32 KiB per lane, so every lane holds several PivCo blocks
    arr = _tile((200, 300), np.uint16)
    _, out = _roundtrip(arr)
    assert np.array_equal(out, arr)


def test_it_compresses_a_smooth_tile():
    arr = _tile((128, 128), np.uint16)
    frame, _ = _roundtrip(arr)
    assert len(frame) < arr.nbytes / 2


@pytest.mark.parametrize("cut", [1, 7, 40])
def test_a_truncated_frame_is_refused(cut):
    frame, _ = _roundtrip(_tile())
    with pytest.raises(RuntimeError):
        geozl.decompress(frame[:-cut])


def test_a_flipped_payload_is_refused_or_decodes_to_the_same_size():
    arr = _tile()
    frame, _ = _roundtrip(arr)
    for at in range(len(frame) - 64, len(frame) - 8, 5):
        bad = bytearray(frame)
        bad[at] ^= 0x5A
        try:
            out = geozl.decompress(bytes(bad))
        except RuntimeError:
            continue
        assert out.nbytes == arr.nbytes


def test_profile_offers_it_beyond_one_byte():
    rows = geozl.profile(_tile((32, 32), np.int16), prior="planar", reps=1)
    assert any(r["graph"] == METHOD for r in rows)
