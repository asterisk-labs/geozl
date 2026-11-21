"""The geozl codec for Zarr format 3."""

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from zarr.codecs import ShardingCodec  # noqa: E402  after importorskip, on purpose
from zarr.registry import get_codec_class  # noqa: E402

from geozl.zarr_codec import GeoZLCodec  # noqa: E402

SMOOTH = "planar>zigzag>transpose>entropy"
INTS = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
FLOATS = ["float16", "float32", "float64"]


def _field(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.indices(shape).sum(axis=0).astype(np.float64)
    values = 50 + 3 * grid + rng.integers(0, 4, shape)
    return values.astype(dtype)


def _array(codec, data, chunks, **kwargs):
    z = zarr.create_array(zarr.storage.MemoryStore(), shape=data.shape,
                          chunks=chunks, dtype=data.dtype, serializer=codec,
                          compressors=None, **kwargs)
    z[...] = data
    return z


def test_zarr_finds_the_codec_by_its_entry_point():
    assert get_codec_class("geozl") is GeoZLCodec


@pytest.mark.parametrize("dtype", INTS + FLOATS)
def test_a_lossless_array_round_trips(dtype):
    recipe = SMOOTH if np.dtype(dtype).itemsize > 1 else "planar>zigzag>entropy"
    data = _field((96, 80), dtype)
    z = _array(GeoZLCodec(recipe=recipe), data, chunks=(32, 48))
    np.testing.assert_array_equal(z[...], data)


@pytest.mark.parametrize("shape, chunks", [
    ((1000,), (300,)),
    ((5, 40, 30), (2, 16, 16)),
    ((3, 4, 10, 12), (2, 3, 5, 12)),
    ((7, 1), (7, 1)),
    ((), ()),
])
def test_every_chunk_rank_round_trips(shape, chunks):
    data = _field(shape, "int16")
    z = _array(GeoZLCodec(recipe=SMOOTH), data, chunks=chunks)
    np.testing.assert_array_equal(z[...], data)


@pytest.mark.parametrize("error, bound", [
    (0.05, lambda x: 0.05),
    ("1%", lambda x: 0.01 * np.abs(x)),
])
def test_a_lossy_array_keeps_its_bound_in_every_chunk(error, bound):
    y, x = np.mgrid[0:128, 0:128]
    data = (np.sin(y / 9) * np.cos(x / 13) * 100 + 200).astype(np.float32)
    # chunks of very different magnitude each resolve their own grid
    data[64:, 64:] *= 50
    z = _array(GeoZLCodec(recipe=SMOOTH, error=error), data, chunks=(32, 32))
    back = z[...]
    assert np.all(np.abs(back.astype(np.float64) - data) <= bound(data))
    assert not np.array_equal(back, data)


def test_nans_survive_a_lossy_array():
    data = _field((64, 64), "float32")
    data[3, :20] = np.nan
    z = _array(GeoZLCodec(recipe=SMOOTH, error=0.5), data, chunks=(32, 32),
               fill_value=np.nan)
    back = z[...]
    np.testing.assert_array_equal(np.isnan(back), np.isnan(data))
    assert np.nanmax(np.abs(back - data)) <= 0.5


def test_a_nodata_sentinel_stays_exact_under_a_lossy_bound():
    data = _field((64, 64), "int16")
    data[10:20, 10:20] = -9999
    z = _array(GeoZLCodec(recipe=SMOOTH, error=3, nodata=-9999), data,
               chunks=(32, 32))
    back = z[...]
    np.testing.assert_array_equal(back == -9999, data == -9999)
    assert np.abs(back.astype(np.int32) - data).max() <= 3


def test_partial_edge_chunks_round_trip():
    data = _field((100, 70), "uint16")
    z = _array(GeoZLCodec(recipe="planar>zigzag>pfor"), data, chunks=(32, 32))
    np.testing.assert_array_equal(z[...], data)
    np.testing.assert_array_equal(z[90:, 60:], data[90:, 60:])


def test_the_codec_serves_as_the_inner_codec_of_a_shard():
    data = _field((128, 128), "uint16")
    shard = ShardingCodec(chunk_shape=(32, 32), codecs=[GeoZLCodec(recipe=SMOOTH)])
    z = zarr.create_array(zarr.storage.MemoryStore(), shape=data.shape,
                          chunks=(64, 64), dtype=data.dtype, serializer=shard,
                          compressors=None)
    z[...] = data
    np.testing.assert_array_equal(z[...], data)
    np.testing.assert_array_equal(z[33:70, 5:99], data[33:70, 5:99])


def test_a_big_endian_array_round_trips():
    data = _field((40, 40), ">u2")
    z = _array(GeoZLCodec(recipe=SMOOTH), data, chunks=(16, 16))
    np.testing.assert_array_equal(z[...], data)


def test_an_array_reopens_from_its_metadata():
    store = zarr.storage.MemoryStore()
    data = _field((64, 64), "float32")
    z = zarr.create_array(store, shape=data.shape, chunks=(32, 32),
                          dtype=data.dtype, compressors=None,
                          serializer=GeoZLCodec(recipe=SMOOTH, error="1%"))
    z[...] = data
    reopened = zarr.open_array(store, mode="r")
    assert reopened.metadata.codecs[0] == GeoZLCodec(recipe=SMOOTH, error="1%")
    np.testing.assert_array_equal(reopened[...], z[...])


@pytest.mark.parametrize("kwargs, configuration", [
    ({"recipe": SMOOTH}, {"recipe": SMOOTH}),
    ({"recipe": SMOOTH, "error": 2}, {"recipe": SMOOTH, "error": "LINEAR:MAX_ERROR=2"}),
    ({"recipe": SMOOTH, "error": "1%"}, {"recipe": SMOOTH, "error": "LOG:MAX_ERROR=1%"}),
    ({"recipe": SMOOTH, "error": 0}, {"recipe": SMOOTH}),
    ({"recipe": SMOOTH, "nodata": float("nan")}, {"recipe": SMOOTH, "nodata": "NaN"}),
    ({"recipe": SMOOTH, "nodata": -9999}, {"recipe": SMOOTH, "nodata": -9999}),
])
def test_the_metadata_holds_the_canonical_configuration(kwargs, configuration):
    codec = GeoZLCodec(**kwargs)
    as_dict = codec.to_dict()
    assert as_dict == {"name": "geozl", "configuration": configuration}
    assert GeoZLCodec.from_dict(json.loads(json.dumps(as_dict))) == codec


@pytest.mark.parametrize("configuration", [
    {},
    {"recipe": ""},
    {"recipe": SMOOTH, "level": 3},
    {"recipe": SMOOTH, "error": 2},
    {"recipe": SMOOTH, "error": "fast"},
    {"recipe": SMOOTH, "nodata": "missing"},
    {"recipe": SMOOTH, "nodata": True},
])
def test_a_bad_configuration_is_refused(configuration):
    with pytest.raises((ValueError, TypeError)):
        GeoZLCodec.from_dict({"name": "geozl", "configuration": configuration})


@pytest.mark.parametrize("dtype", ["bool", "complex64"])
def test_an_unsupported_dtype_is_refused_when_made(dtype):
    with pytest.raises(ValueError, match="does not support dtype"):
        zarr.create_array(zarr.storage.MemoryStore(), shape=(8, 8), chunks=(4, 4),
                          dtype=dtype, serializer=GeoZLCodec(recipe=SMOOTH),
                          compressors=None)


@pytest.mark.parametrize("dtype, kwargs", [
    ("float32", {"recipe": "planar>zigzag>entropy"}),   # plain entropy above 2 bytes
    ("uint16", {"recipe": "no>such>recipe"}),
    ("uint16", {"recipe": SMOOTH, "nodata": -9999}),   # does not fit uint16
    ("float16", {"recipe": SMOOTH, "nodata": -9999}),  # no float16 sentinels
])
def test_a_configuration_the_encoder_cannot_use_is_refused_at_the_first_write(dtype, kwargs):
    z = zarr.create_array(zarr.storage.MemoryStore(), shape=(8, 8), chunks=(4, 4),
                          dtype=dtype, serializer=GeoZLCodec(**kwargs), compressors=None)
    with pytest.raises(ValueError, match="cannot encode"):
        z[...] = 1


def test_an_array_reads_when_this_geozl_cannot_build_its_recipe():
    # Decoding needs only the frame, so a recipe a later release dropped, or a
    # graph composed by another encoder, must not make the array unreadable.
    store = zarr.storage.MemoryStore()
    data = _field((32, 32), "uint16")
    _array_in(store, data)
    meta = json.loads(store._store_dict["zarr.json"].to_bytes())
    meta["codecs"][0]["configuration"]["recipe"] = "a>recipe>from>elsewhere"
    store._store_dict["zarr.json"] = store._store_dict["zarr.json"].from_bytes(
        json.dumps(meta).encode())
    np.testing.assert_array_equal(zarr.open_array(store, mode="r")[...], data)


def _array_in(store, data):
    z = zarr.create_array(store, shape=data.shape, chunks=(16, 16), dtype=data.dtype,
                          serializer=GeoZLCodec(recipe=SMOOTH), compressors=None)
    z[...] = data
    return z


def test_a_frame_larger_than_its_chunk_is_refused():
    store = zarr.storage.MemoryStore()
    big = _array(GeoZLCodec(recipe=SMOOTH), _field((32, 32), "uint16"), chunks=(32, 32))
    small = zarr.create_array(store, shape=(8, 8), chunks=(8, 8), dtype="uint16",
                              serializer=GeoZLCodec(recipe=SMOOTH), compressors=None)
    frame = big.store_path.store._store_dict["c/0/0"]
    store._store_dict["c/0/0"] = frame
    with pytest.raises(ValueError, match="allowed"):
        small[...]
