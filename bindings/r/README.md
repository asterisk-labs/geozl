# geozl for R

R bindings to GeoZL. They have the same graph, compress, decompress and profile
calls as the Python package and write the same frames.

```r
library(geozl)

tile <- outer(1:1024, 1:1024, function(x, y) 2000 + 5 * (x - 1) + 8 * (y - 1))

print(geozl_profile(tile, datatype = "uint16", reps = 1))

g <- geozl_graph(tile, "planar>zigzag>pfor", datatype = "uint16")
frame <- geozl_compress(tile, g)
back <- geozl_decompress(frame, "uint16", dim = dim(tile))
identical(back, tile)
```

## Install

The package compiles GeoZL and OpenZL from source with their own CMake files,
so it needs CMake 3.21 or newer and C and C++ compilers.

```r
remotes::install_github("asterisk-labs/geozl", subdir = "bindings/r")
```

From a checkout, `R CMD INSTALL bindings/r` builds the repository's `core/`
and `extern/openzl/` in place. Anywhere else the package builds the copy it
carries in `src/ext/`. That copy is kept in git, as taco and cozip keep theirs;
`make r-vendor` refreshes it after a change to the core, and a release fails
when it is out of date.

## Rasters

- **Datatypes.** R holds rasters as double or integer, as terra and stars do.
  `datatype` names the storage type, as numpy (`"uint16"`), terra (`"INT2U"`)
  or GDAL (`"UInt16"`) spell it, for 8 to 64 bit integers and 16 to 64 bit
  floats. Integer types refuse values that are not whole or do not fit; float
  types round to nearest. `geozl_decompress` returns double by default,
  `as = "integer"` for integer types, or `as = "raw"` for the native bytes.
- **Layout.** The fastest-varying dimension is the row. An array shaped
  `(x, y)` or `(x, y, band)`, as stars holds a raster, gives the same frame as
  Python does for `(y, x)` or `(band, y, x)`. For `v <- terra::values(r)`, pass
  `width = terra::ncol(r), planes = terra::nlyr(r)` because `v` is shaped
  `(ncell, nlyr)`. `terra::as.array()` is `(y, x, band)`; pass it through
  `aperm(a, c(2, 1, 3))` first.
- **NoData.** `nodata = -9999` stores `NA` as that sentinel, and
  `geozl_decompress(..., nodata = -9999)` turns it back into `NA`. A float
  raster with missing values and no sentinel uses NaN handling, which keeps
  `NA` and `NaN` apart in a lossless frame; a lossy one writes every hole back
  with the first one's bits.
- **Bounded error.** `error = 0.5` bounds the absolute error, `error = "1%"`
  the relative error, and full `LINEAR`, `LOG` and `SQRT` recipes are accepted,
  as in Python.
- **torch.** torch for R has no uint16, uint32 or uint64 tensors. Read other
  types without a copy with
  `torch_tensor_from_buffer(geozl_decompress(frame, as = "raw"), shape, dtype)`;
  the bytes are in C order, so `shape` is `(band, y, x)`.

A frame does not record its datatype or shape. Keep them beside it, as Rumi
does.

## Development

```sh
make r          # roxygen2, then the tests from the checkout
make r-check    # vendor, build and R CMD check the source package
```

The tests decode every golden frame when they find
`bindings/python/test/golden` or `GEOZL_GOLDEN_DIR`, and compare frames with
Python's when `GEOZL_PYTHON` names a Python with geozl, which `make r` sets.
