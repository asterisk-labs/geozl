test_that("width is the fastest dimension and planes the slowest", {
  x <- round(smooth_raster(48, 32, 0, 4000))
  g <- geozl_graph(x, "planar>zigzag>pfor", datatype = "uint16")
  expect_identical(g$width, 48)
  expect_identical(g$planes, 1)
  explicit <- geozl_graph(as.vector(x), "planar>zigzag>pfor", width = 48,
                          datatype = "uint16")
  expect_identical(geozl_compress(x, g), geozl_compress(as.vector(x), explicit))

  cube <- array(round(smooth_raster(16, 12 * 3, 0, 4000)), c(16, 12, 3))
  gc <- geozl_graph(cube, "planar>zigzag>pfor", datatype = "uint16")
  expect_identical(c(gc$width, gc$planes), c(16, 3))
  back <- geozl_decompress(geozl_compress(cube, gc), "uint16", dim = dim(cube))
  expect_identical(back, cube)
})

test_that("geometry is checked", {
  expect_error(geozl_graph(1:64, "id>zstd"), "give `width`")
  expect_error(geozl_graph(1:64, "id>zstd", width = 8, planes = 3),
               "do not split")
  expect_error(geozl_graph(1:64, "id>zstd", width = 0), "whole number")
  expect_error(geozl_decompress(geozl_compress(1:64, geozl_graph(1:64,
               "id>zstd", width = 8)), "int32", dim = c(5, 5)), "holds 25")
})
