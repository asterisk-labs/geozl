x <- round(smooth_raster(32, 32, 0, 4000))
g <- geozl_graph(x, "planar>zigzag>pfor", datatype = "uint16")

test_that("coefficients round trip beside the frame", {
  coeffs <- list(1:3, c(-5L, 7L), 2147483647L)
  frame <- geozl_compress(x, g, coeffs = coeffs)
  expect_identical(geozl_coeffs(frame), coeffs)
  expect_identical(geozl_decompress(frame, "uint16", dim = dim(x)), x)
  expect_identical(geozl_coeffs(geozl_compress(x, g, coeffs = list(c(2, 4)))),
                   list(c(2L, 4L)))
})

test_that("a frame without coefficients has none", {
  expect_null(geozl_coeffs(geozl_compress(x, g)))
})

test_that("invalid coefficients are refused", {
  expect_error(geozl_compress(x, g, coeffs = 1:3), "list")
  expect_error(geozl_compress(x, g, coeffs = list()), "at least one")
  expect_error(geozl_compress(x, g, coeffs = list(integer())), "empty")
  expect_error(geozl_compress(x, g, coeffs = list(1.5)), "whole number")
  expect_error(geozl_compress(x, g, coeffs = list(NA_integer_)), "whole number")
  expect_error(geozl_compress(x, g, coeffs = list(2^31)), "int32")
  expect_error(geozl_compress(x, g, coeffs = rep(list(1L), 256)), "at most 255")
  expect_error(geozl_compress(x, g, coeffs = list(1:3000)), "byte limit")
})
