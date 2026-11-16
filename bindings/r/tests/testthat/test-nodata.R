test_that("NA in an integer raster round trips through a sentinel", {
  x <- round(smooth_raster(40, 30, -2000, 2000))
  x[c(5, 77, 1200)] <- NA
  g <- geozl_graph(x, "planar>zigzag>pfor", nodata = -9999, dtype = "INT2S")
  frame <- geozl_compress(x, g)
  back <- geozl_decompress(frame, "INT2S", dim = dim(x), nodata = -9999)
  expect_identical(back, x)
  raw_values <- geozl_decompress(frame, "int16", dim = dim(x))
  expect_identical(raw_values[c(5, 77, 1200)], rep(-9999, 3))
})

test_that("NaN in a float raster selects NaN nodata", {
  x <- smooth_raster(40, 30, 0, 1)
  x[c(2, 300)] <- NaN
  x[400] <- NA
  g <- geozl_graph(x, "planar>zigzag>transpose>entropy", dtype = "float64")
  back <- geozl_decompress(geozl_compress(x, g), "float64", dim = dim(x))
  # Lossless float64 keeps NA and NaN apart, payloads and all.
  expect_identical(back, x)
  expect_true(is.na(back[400]) && !is.nan(back[400]))
  expect_true(all(is.nan(back[c(2, 300)])))

  # A quantizer cannot carry a NaN, so a lossy frame masks every hole and
  # writes them back with the first payload.
  lossy <- geozl_graph(x, "planar>zigzag>transpose>entropy", error = 0.01,
                       dtype = "float32")
  back <- geozl_decompress(geozl_compress(x, lossy), "float32", dim = dim(x))
  expect_identical(which(is.na(back)), which(is.na(x)))
  expect_lte(max(abs(back - x), na.rm = TRUE), 0.01 + 1e-6)
})

test_that("float32 and float16 store NA and NaN as one NaN", {
  x <- matrix(c(NA, NaN, 1:14), 4, 4)
  for (dtype in c("float32", "float16")) {
    g <- geozl_graph(x, "id>zstd", dtype = dtype)
    back <- geozl_decompress(geozl_compress(x, g), dtype, dim = dim(x))
    expect_true(all(is.nan(back[1:2])))
  }
})

test_that("a sentinel takes NaN as well as NA", {
  x <- matrix(c(NA, NaN, 1:14), 4, 4)
  g <- geozl_graph(x, "id>zstd", nodata = -9999, dtype = "float32")
  frame <- geozl_compress(x, g)
  raw_values <- geozl_decompress(frame, "float32", dim = dim(x))
  expect_identical(raw_values[1:2], rep(-9999, 2))
})

test_that("a float sentinel maps NA both ways", {
  x <- smooth_raster(20, 20, 0, 100)
  x[c(3, 50)] <- NA
  g <- geozl_graph(x, "id>zstd", nodata = -1, dtype = "FLT4S")
  back <- geozl_decompress(geozl_compress(x, g), "FLT4S", dim = dim(x),
                           nodata = -1)
  expect_identical(which(is.na(back)), c(3L, 50L))
})

test_that("float sentinels are matched at their storage type", {
  x <- matrix(c(NA, seq(0.2, 1.6, length.out = 15)), 4, 4)
  g <- geozl_graph(x, "id>zstd", nodata = 0.1, dtype = "float32")
  back <- geozl_decompress(geozl_compress(x, g), "float32", dim = dim(x),
                           nodata = 0.1)
  expect_true(is.na(back[1]))
  expect_false(anyNA(back[-1]))
})

test_that("negative zero sentinel does not hide positive zero", {
  x <- matrix(c(NA, 0, 1:14), 4, 4)
  g <- geozl_graph(x, "id>zstd", nodata = -0, dtype = "float64")
  back <- geozl_decompress(geozl_compress(x, g), "float64", dim = dim(x),
                           nodata = -0)
  expect_true(is.na(back[1]))
  expect_false(is.na(back[2]))
  expect_identical(1 / back[2], Inf)
})

test_that("float16 has no sentinel", {
  expect_error(geozl_graph(matrix(1, 8, 8), "id>zstd", nodata = -1,
                           dtype = "float16"),
               "a half float carries no sentinel, NaN still works on one",
               fixed = TRUE)
})

test_that("NaN nodata reads every NaN of a float dtype as NA", {
  x <- matrix(c(NaN, NA, -1.5, 0, 2.25, 1:11), 4, 4)
  for (dtype in c("float16", "float32", "float64")) {
    frame <- geozl_compress(x, geozl_graph(x, "id>zstd", dtype = dtype))
    plain <- geozl_decompress(frame, dtype, dim = dim(x))
    for (nodata in list(NaN, NA)) {
      back <- geozl_decompress(frame, dtype, dim = dim(x), nodata = nodata)
      expect_identical(back[1:2], c(NA_real_, NA_real_), label = dtype)
      expect_false(any(is.nan(back)), label = dtype)
      expect_identical(back[-(1:2)], plain[-(1:2)], label = dtype)
    }
  }
})
