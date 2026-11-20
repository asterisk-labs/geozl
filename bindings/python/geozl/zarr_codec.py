"""The ``geozl`` codec for Zarr format 3.

An array -> bytes codec that stores each chunk as one GeoZL frame. Zarr finds it
through the ``zarr.codecs`` entry point, so an installed geozl is all a reader
needs. The specification lives in the zarr-extensions registry under
``codecs/geozl``.

.. code-block:: python

    import zarr
    from geozl.zarr_codec import GeoZLCodec

    z = zarr.create_array(
        store, shape=(4096, 4096), chunks=(512, 512), dtype="uint16",
        serializer=GeoZLCodec(recipe="planar>zigzag>transpose>entropy"),
        compressors=None,
    )

A graph is built per chunk from the chunk itself. That is cheap once the
library is loaded, keeps the codec thread-safe (a ``geozl.Graph`` is not), and
resolves a lossy grid from the samples it has to hold, so every chunk keeps the
declared bound.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import JSON, parse_named_configuration

from ._2d import _normalize_error, compress, decompress, graph
from ._dtype import refuse_non_number

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

__all__ = ["CODEC_NAME", "GeoZLCodec"]

CODEC_NAME = "geozl"

# The dtypes the specification lists. GeoZL itself takes any 1, 2, 4 or 8 byte
# element losslessly, but the quantizers and NoData need a numeric type.
_SUPPORTED = frozenset(np.dtype(name) for name in (
    "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
    "float16", "float32", "float64"))

# JSON has no non-finite numbers. These are the spellings Zarr uses for fill
# values, so a configuration reads the same way as the array metadata around it.
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def _parse_recipe(recipe: object) -> str:
    if not isinstance(recipe, str) or not recipe:
        raise ValueError(f"recipe must be a non-empty string, got {recipe!r}")
    if "\x00" in recipe:
        raise ValueError("recipe must not contain a NUL character")
    return recipe


def _parse_error(error: object) -> str | None:
    """Return the canonical error recipe, or None for lossless."""
    # The Python shorthands (2, 0.5, "1%") expand to the recipe the metadata
    # stores, so another implementation never has to know them.
    return _normalize_error(error)  # type: ignore[arg-type]


def _parse_nodata(nodata: object) -> int | float | str | None:
    """Return the JSON form of a NoData sentinel."""
    if nodata is None:
        return None
    if isinstance(nodata, str):
        if nodata not in _NONFINITE:
            raise ValueError(f"nodata must be a number, one of "
                             f"{sorted(_NONFINITE)} or null, got {nodata!r}")
        return nodata
    refuse_non_number(nodata)
    if isinstance(nodata, (int, np.integer)):
        return int(nodata)
    value = float(nodata)  # type: ignore[arg-type]
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _nodata_value(nodata: int | float | str | None) -> int | float | None:
    if isinstance(nodata, str):
        return _NONFINITE[nodata]
    return nodata


def _native(dtype: np.dtype[Any]) -> np.dtype[Any]:
    return dtype.newbyteorder("=")


def _check_dtype(dtype: np.dtype[Any]) -> None:
    if _native(dtype) not in _SUPPORTED:
        raise ValueError(f"the geozl codec does not support dtype {dtype}; it "
                         f"takes {', '.join(sorted(str(d) for d in _SUPPORTED))}")


def _raster(chunk: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """View a chunk in the geometry the specification assigns it.

    Rows are the last axis. A chunk of three or more axes is a stack of
    ``prod(shape[:-2])`` planes of ``shape[-2]`` rows; one of fewer axes is a
    single plane. An empty chunk is one row of nothing.
    """
    if chunk.size == 0:
        return chunk.reshape(1, 0)
    if chunk.ndim == 0:
        return chunk.reshape(1, 1)
    if chunk.ndim == 1:
        return chunk.reshape(1, -1)
    if chunk.ndim == 2:
        return chunk
    return chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])


@dataclass(frozen=True)
class GeoZLCodec(ArrayBytesCodec):
    """Compress each chunk with a GeoZL recipe.

    Parameters
    ----------
    recipe
        A GeoZL graph recipe, such as ``"planar>zigzag>transpose>entropy"``.
        ``geozl.profile`` ranks recipes on a sample.
    error
        ``None`` for lossless, or a bound: a positive number for ``LINEAR``, a
        percentage string such as ``"1%"`` for ``LOG``, or a full ``LINEAR``,
        ``LOG`` or ``SQRT`` recipe. The metadata stores the full recipe.
    nodata
        A NoData sentinel kept exact through the lossy stages, or ``None``.
        NaNs in floating-point chunks are always kept.
    """

    is_fixed_size = False

    recipe: str
    error: str | None
    nodata: int | float | str | None

    def __init__(self, *, recipe: str, error: Any = None, nodata: Any = None) -> None:
        object.__setattr__(self, "recipe", _parse_recipe(recipe))
        object.__setattr__(self, "error", _parse_error(error))
        object.__setattr__(self, "nodata", _parse_nodata(nodata))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        unknown = set(configuration) - {"recipe", "error", "nodata"}
        if unknown:
            raise ValueError(f"unknown geozl codec configuration keys: "
                             f"{sorted(unknown)}")
        if "recipe" not in configuration:
            raise ValueError("the geozl codec configuration needs a recipe")
        error = configuration.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"error must be a recipe string or null in the "
                             f"metadata, got {error!r}")
        return cls(**configuration)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        configuration: dict[str, JSON] = {"recipe": self.recipe}
        if self.error is not None:
            configuration["error"] = self.error
        if self.nodata is not None:
            configuration["nodata"] = self.nodata
        return {"name": CODEC_NAME, "configuration": configuration}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: Any,  # its class differs between Zarr releases
    ) -> None:
        # Zarr validates on open as well as on create. Decoding needs only the
        # frame and the dtype, so check nothing else here: an array whose recipe
        # this geozl cannot build (one a later release dropped, or a graph
        # another encoder composed) must stay readable. A recipe, error or
        # sentinel the encoder cannot use is refused at the first write.
        _check_dtype(dtype.to_native_dtype())

    def _encode_array(self, chunk: np.ndarray[Any, Any]) -> bytes:
        raster = _raster(np.ascontiguousarray(chunk))
        try:
            g = graph(raster, self.recipe, error=self.error,
                      nodata=_nodata_value(self.nodata))
            return compress(raster, graph=g)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"the geozl codec cannot encode a {raster.dtype} chunk "
                             f"with {self.to_dict()['configuration']}: {exc}") from exc

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        array = chunk_array.as_numpy_array()
        native = _native(array.dtype)
        if array.dtype != native:
            array = array.astype(native)
        return chunk_spec.prototype.buffer.from_bytes(self._encode_array(array))

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        dtype = chunk_spec.dtype.to_native_dtype()
        native = _native(dtype)
        expected = math.prod(chunk_spec.shape) * native.itemsize
        # The frame is untrusted: refuse one that declares more output than the
        # chunk holds before anything is allocated for it.
        raw = decompress(chunk_bytes.to_bytes(), verify=True,
                         max_output_size=expected)
        if raw.size != expected:
            raise ValueError(f"a geozl frame decoded to {raw.size} bytes, the "
                             f"chunk {chunk_spec.shape} of {dtype} holds {expected}")
        array = raw.view(native).reshape(chunk_spec.shape)
        if native != dtype:
            array = array.astype(dtype)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(array)

    async def _encode_single(self, chunk_array: NDBuffer,
                             chunk_spec: ArraySpec) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_array, chunk_spec)

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
