# Each dtype with values that fit it exactly, extremes included.
cases <- list(
  uint8 = function(x) round(x * 255),
  uint16 = function(x) round(x * 65535),
  uint32 = function(x) round(x * 4294967295),
  uint64 = function(x) round(x * 2^53),
  int8 = function(x) round(x * 255) - 128,
  int16 = function(x) round(x * 65535) - 32768,
  int32 = function(x) round(x * 4294967294) - 2147483647,
  int64 = function(x) round(x * 2^54) - 2^53,
  float16 = function(x) round(x * 4096) - 2048,
  float32 = function(x) (round(x * 2^20) - 2^19) / 1024,
  float64 = function(x) x * 1e300 - 5e299
)

base <- smooth_raster(40, 24, 0, 1)

for (name in names(cases)) {
  test_that(sprintf("%s round trips through id>zstd", name), {
    x <- cases[[name]](base)
    g <- geozl_graph(x, "id>zstd", dtype = name)
    back <- geozl_decompress(geozl_compress(x, g), name, dim = dim(x))
    expect_identical(back, x)
  })
}

test_that("one and two byte integers round trip through the fused codecs", {
  for (name in c("uint8", "int8", "uint16", "int16")) {
    x <- cases[[name]](base)
    for (method in c("planar>zigzag>pfor", "planar>zigzag>lanes",
                     "planar>zigzag>entropy", "med>zigzag>entropy")) {
      g <- geozl_graph(x, method, dtype = name)
      back <- geozl_decompress(geozl_compress(x, g), name, dim = dim(x))
      expect_identical(back, x, label = paste(name, method))
    }
  }
})

test_that("wide integers round trip through transpose>entropy", {
  for (name in c("uint32", "int32", "uint64", "int64")) {
    x <- cases[[name]](base)
    g <- geozl_graph(x, "planar>zigzag>transpose>entropy", dtype = name)
    back <- geozl_decompress(geozl_compress(x, g), name, dim = dim(x))
    expect_identical(back, x, label = name)
  }
})

test_that("integer input and output stay integer", {
  x <- matrix(as.integer(cases$uint16(base)), nrow(base))
  g <- geozl_graph(x, "planar>zigzag>pfor", dtype = "INT2U")
  back <- geozl_decompress(geozl_compress(x, g), "UInt16", dim = dim(x),
                           as = "integer")
  expect_identical(back, x)
})

test_that("raw input is native samples and raw output returns them", {
  x <- cases$int16(base)
  g <- geozl_graph(x, "planar>zigzag>pfor", dtype = "int16")
  frame <- geozl_compress(x, g)
  bytes <- geozl_decompress(frame)
  expect_type(bytes, "raw")
  expect_length(bytes, 2 * length(x))
  expect_identical(geozl_decompress(frame, "int16", as = "raw"), bytes)
  g_raw <- geozl_graph(bytes, "planar>zigzag>pfor", width = nrow(x),
                       dtype = "int16")
  expect_identical(geozl_compress(bytes, g_raw), frame)
})

test_that("a frame with no samples decodes to empty raw and typed output", {
  x <- matrix(integer(), 8, 0)
  g <- geozl_graph(x, "id>zstd", dtype = "int16")
  frame <- geozl_compress(x, g)
  expect_identical(geozl_decompress(frame, max_output_size = 0), raw())
  expect_identical(geozl_decompress(frame, verify = FALSE), raw())
  expect_identical(geozl_decompress(frame, "int16", dim = dim(x),
                                    as = "integer"), x)
})

test_that("an empty first dimension gives a width of 0, as in Python", {
  x <- matrix(integer(), 0, 8)
  g <- geozl_graph(x, "id>zstd", dtype = "int16")
  expect_identical(c(g$width, g$planes), c(0, 1))
  back <- geozl_decompress(geozl_compress(x, g), "int16", dim = dim(x),
                           as = "integer")
  expect_identical(back, x)
  expect_gt(nrow(geozl_profile(x, dtype = "int16", reps = 1)), 0)
})

test_that("terra and GDAL datatype names map to the same storage", {
  pairs <- list(c("INT1U", "Byte", "uint8"), c("INT2S", "Int16", "int16"),
                c("INT4U", "UInt32", "uint32"), c("INT8S", "Int64", "int64"),
                c("FLT4S", "Float32", "float32"), c("FLT8S", "Float64", "float64"))
  for (p in pairs) {
    expect_identical(geozl:::.dtype(p[[1]])$name, p[[3]])
    expect_identical(geozl:::.dtype(p[[2]])$name, p[[3]])
  }
  expect_identical(geozl:::.dtype("Float16")$name, "float16")
})

test_that("float16 rounds to nearest even and keeps exact values", {
  x <- c(0.5, -1.5, 65504, -65504, 2^-24, 2^-14, 0, 1/3, 2049)
  g <- geozl_graph(x, "id>zstd", width = length(x), dtype = "float16")
  back <- geozl_decompress(geozl_compress(x, g), "float16")
  expect_identical(back, c(0.5, -1.5, 65504, -65504, 2^-24, 2^-14, 0,
                           0.333251953125, 2048))
})

test_that("the dtype defaults from the R type", {
  expect_identical(geozl_graph(matrix(1L, 8, 8), "id>zstd")$dtype, "int32")
  expect_identical(geozl_graph(matrix(1, 8, 8), "id>zstd")$dtype, "float64")
  expect_identical(geozl_graph(as.raw(1:64), "id>zstd", width = 8)$dtype,
                   "uint8")
})

test_that("a graph keeps its fields and prints them", {
  g <- geozl_graph(matrix(1, 48, 4), "planar>zigzag>pfor", dtype = "UInt16")
  expect_named(g, c("ptr", "method", "width", "planes", "dtype", "itemsize",
                    "error", "nodata"))
  expect_identical(g$itemsize, 2)
  expect_output(print(g), paste("<geozl_graph> planar>zigzag>pfor, uint16,",
                                "row width 48, 1 plane, lossless"), fixed = TRUE)
})
