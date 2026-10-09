# Each datatype with values that fit it exactly, extremes included.
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
    g <- geozl_graph(x, "id>zstd", datatype = name)
    back <- geozl_decompress(geozl_compress(x, g), name, dim = dim(x))
    expect_identical(back, x)
  })
}

test_that("one and two byte integers round trip through the fused codecs", {
  for (name in c("uint8", "int8", "uint16", "int16")) {
    x <- cases[[name]](base)
    for (method in c("planar>zigzag>pfor", "planar>zigzag>pivco",
                     "planar>zigzag>entropy", "med>zigzag>entropy")) {
      g <- geozl_graph(x, method, datatype = name)
      back <- geozl_decompress(geozl_compress(x, g), name, dim = dim(x))
      expect_identical(back, x, label = paste(name, method))
    }
  }
})

test_that("wide integers round trip through transpose>entropy", {
  for (name in c("uint32", "int32", "uint64", "int64")) {
    x <- cases[[name]](base)
    g <- geozl_graph(x, "planar>zigzag>transpose>entropy", datatype = name)
    back <- geozl_decompress(geozl_compress(x, g), name, dim = dim(x))
    expect_identical(back, x, label = name)
  }
})

test_that("integer input and output stay integer", {
  x <- matrix(as.integer(cases$uint16(base)), nrow(base))
  g <- geozl_graph(x, "planar>zigzag>pfor", datatype = "INT2U")
  back <- geozl_decompress(geozl_compress(x, g), "UInt16", dim = dim(x),
                           as = "integer")
  expect_identical(back, x)
})

test_that("raw input is native samples and raw output returns them", {
  x <- cases$int16(base)
  g <- geozl_graph(x, "planar>zigzag>pfor", datatype = "int16")
  frame <- geozl_compress(x, g)
  bytes <- geozl_decompress(frame)
  expect_type(bytes, "raw")
  expect_length(bytes, 2 * length(x))
  expect_identical(geozl_decompress(frame, "int16", as = "raw"), bytes)
  g_raw <- geozl_graph(bytes, "planar>zigzag>pfor", width = nrow(x),
                       datatype = "int16")
  expect_identical(geozl_compress(bytes, g_raw), frame)
})

test_that("terra and GDAL datatype names map to the same storage", {
  pairs <- list(c("INT1U", "Byte", "uint8"), c("INT2S", "Int16", "int16"),
                c("INT4U", "UInt32", "uint32"), c("INT8S", "Int64", "int64"),
                c("FLT4S", "Float32", "float32"), c("FLT8S", "Float64", "float64"))
  for (p in pairs) {
    expect_identical(geozl:::.datatype(p[[1]])$name, p[[3]])
    expect_identical(geozl:::.datatype(p[[2]])$name, p[[3]])
  }
  expect_identical(geozl:::.datatype("Float16")$name, "float16")
})

test_that("float16 rounds to nearest even and keeps exact values", {
  x <- c(0.5, -1.5, 65504, -65504, 2^-24, 2^-14, 0, 1/3, 2049)
  g <- geozl_graph(x, "id>zstd", width = length(x), datatype = "float16")
  back <- geozl_decompress(geozl_compress(x, g), "float16")
  expect_identical(back, c(0.5, -1.5, 65504, -65504, 2^-24, 2^-14, 0,
                           0.333251953125, 2048))
})

test_that("the datatype defaults from the R type", {
  expect_identical(geozl_graph(matrix(1L, 8, 8), "id>zstd")$datatype, "int32")
  expect_identical(geozl_graph(matrix(1, 8, 8), "id>zstd")$datatype, "float64")
  expect_identical(geozl_graph(as.raw(1:64), "id>zstd", width = 8)$datatype,
                   "uint8")
})
