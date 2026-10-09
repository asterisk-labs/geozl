test_that("error expands like the Python API", {
  normalize <- geozl:::.normalize_error
  expect_null(normalize(NULL))
  expect_null(normalize(0))
  expect_null(normalize("0%"))
  expect_null(normalize("LINEAR:MAX_ERROR=0"))
  expect_null(normalize("LOG:MAX_ERROR=0.0%"))
  expect_identical(normalize(2), "LINEAR:MAX_ERROR=2")
  expect_identical(normalize(0.5), "LINEAR:MAX_ERROR=0.5")
  expect_identical(normalize(1/3), "LINEAR:MAX_ERROR=0.3333333333333333")
  expect_identical(normalize(1e-5), "LINEAR:MAX_ERROR=1e-05")
  expect_identical(normalize("1%"), "LOG:MAX_ERROR=1%")
  expect_identical(normalize("SQRT:MAX_ERROR=2N"), "SQRT:MAX_ERROR=2N")
  expect_error(normalize(-1), "negative")
  expect_error(normalize(Inf), "finite")
  expect_error(normalize(TRUE), "logical")
  expect_error(normalize("100%"), "between")
  expect_error(normalize("abc"), "LINEAR, LOG or SQRT")
})

x <- smooth_raster(64, 48, 200, 3000) +
  matrix(sin(seq_len(64 * 48)) * 3, 64, 48)

test_that("an absolute bound holds on every sample", {
  g <- geozl_graph(x, "planar>zigzag>transpose>entropy", error = 0.5,
                   datatype = "float32")
  expect_identical(g$error, "LINEAR:MAX_ERROR=0.5")
  back <- geozl_decompress(geozl_compress(x, g), "float32", dim = dim(x))
  ref <- geozl_decompress(geozl_compress(x, geozl_graph(x, "id>zstd",
                          datatype = "float32")), "float32", dim = dim(x))
  expect_lte(max(abs(back - ref)), 0.5)
})

test_that("a relative bound holds on every sample", {
  g <- geozl_graph(x, "planar>zigzag>transpose>entropy", error = "1%",
                   datatype = "float64")
  back <- geozl_decompress(geozl_compress(x, g), "float64", dim = dim(x))
  expect_true(all(abs(back - x) <= 0.01 * abs(x)))
})

test_that("tiles outside the graph's domain are refused", {
  g <- geozl_graph(x, "planar>zigzag>transpose>entropy", error = 0.5,
                   datatype = "float32")
  expect_error(geozl_compress(x * 1e6, g), "geozl_compress failed")
})
