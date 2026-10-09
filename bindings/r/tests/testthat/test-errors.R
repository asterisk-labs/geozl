tile <- round(smooth_raster(32, 32, 0, 4000))

test_that("values that do not fit the datatype are refused", {
  expect_error(geozl_graph(tile + 70000, "id>zstd", datatype = "uint16"),
               "does not fit uint16")
  expect_error(geozl_graph(tile + 0.5, "id>zstd", datatype = "int32"),
               "does not fit int32")
  expect_error(geozl_graph(tile * 1e40, "id>zstd", datatype = "float32"),
               "float32 range")
  expect_error(geozl_graph(tile * 100, "id>zstd", datatype = "float16"),
               "float16 range")
  expect_error(geozl_graph(tile, "id>zstd", datatype = "uint12"),
               "unknown datatype")
  expect_error(geozl_graph(matrix(TRUE, 8, 8), "id>zstd"), "logical")
})

test_that("NA in an integer raster needs nodata", {
  x <- tile
  x[3] <- NA
  expect_error(geozl_graph(x, "id>zstd", datatype = "uint16"),
               "needs `nodata`")
  expect_error(geozl_graph(tile, "id>zstd", nodata = NaN, datatype = "uint16"),
               "cannot use NaN")
  expect_error(geozl_graph(tile, "id>zstd", nodata = 70000, datatype = "uint16"),
               "nodata 70000 does not fit")
  expect_error(geozl_graph(tile, "id>zstd", nodata = c(1, 2)), "single number")
})

test_that("bad recipes and graphs fail with a reason", {
  expect_error(geozl_graph(tile, "planar>nothing", datatype = "uint16"),
               "geozl_graph failed")
  expect_error(geozl_graph(tile, "", datatype = "uint16"), "recipe")
  expect_error(geozl_compress(tile, list()), "geozl_graph\\(\\)")
  g <- geozl_graph(tile, "id>zstd", datatype = "uint16")
  restored <- unserialize(serialize(g, NULL))
  expect_error(geozl_compress(tile, restored), "no longer open")
})

test_that("damaged and oversized frames are refused", {
  g <- geozl_graph(tile, "planar>zigzag>pfor", datatype = "uint16")
  frame <- geozl_compress(tile, g)
  expect_error(geozl_decompress(frame[seq_len(length(frame) %/% 2)]))
  expect_error(geozl_decompress(as.raw(0:63)), "unreadable frame")
  expect_error(geozl_decompress(as.integer(frame)), "raw vector")
  expect_error(geozl_decompress(frame, max_output_size = 100), "above")
  flipped <- frame
  flipped[length(flipped) - 4] <- xor(flipped[length(flipped) - 4], as.raw(1))
  expect_error(geozl_decompress(flipped))
  expect_error(geozl_decompress(frame, verify = NA), "TRUE or FALSE")
})

test_that("integer output needs values R can hold", {
  x <- matrix(4294967295, 4, 4)
  frame <- geozl_compress(x, geozl_graph(x, "id>zstd", datatype = "uint32"))
  expect_error(geozl_decompress(frame, "uint32", as = "integer"),
               "does not fit an R integer")
  expect_identical(geozl_decompress(frame, "uint32"), rep(4294967295, 16))
  f32 <- geozl_compress(tile, geozl_graph(tile, "id>zstd", datatype = "float32"))
  expect_error(geozl_decompress(f32, "float32", as = "integer"),
               "as = \"double\"")
  expect_error(geozl_decompress(f32, as = "double"), "`datatype` is needed")
})

test_that("64-bit values above 2^53 warn when read as double", {
  x <- c(2^60, 1)
  frame <- geozl_compress(x, geozl_graph(x, "id>zstd", width = 2,
                                         datatype = "uint64"))
  expect_warning(back <- geozl_decompress(frame, "uint64"), "2\\^53")
  expect_identical(back, x)
})
