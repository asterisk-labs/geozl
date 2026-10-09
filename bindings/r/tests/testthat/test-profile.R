test_that("profile ranks the recipes a prior allows", {
  x <- round(smooth_raster(64, 64, 0, 4000))
  p <- geozl_profile(x, datatype = "uint16", reps = 1)
  expect_s3_class(p, "geozl_profile")
  expect_named(p, c("graph", "bytes", "ratio", "encode_mbps", "decode_mbps",
                    "shannon_pct"))
  expect_gt(nrow(p), 5)
  expect_true(all(diff(p$ratio) <= 0))
  expect_true(all(startsWith(p$graph, "planar>") | startsWith(p$graph, "id>")))
  expect_true("planar>zigzag>pfor" %in% p$graph)
  best <- geozl_graph(x, p$graph[[1]], datatype = "uint16")
  expect_identical(length(geozl_compress(x, best)), as.integer(p$bytes[[1]]))
  expect_output(print(p), "dec MB/s")
  expect_error(geozl_profile(x, prior = "bogus", datatype = "uint16"),
               "not one of")
  expect_error(geozl_profile(x, prior = "", datatype = "uint16"),
               "non-empty")
  expect_error(geozl_profile(x, reps = 0), "whole number")
  expect_error(geozl_profile(x, reps = Inf), "whole number")
  expect_error(geozl_profile(x, reps = .Machine$integer.max + 1), "at most")
})

test_that("a prior of NULL tries every predictor", {
  x <- round(smooth_raster(32, 32, 0, 255))
  p <- geozl_profile(x, prior = NULL, datatype = "uint8", reps = 1)
  expect_true(any(startsWith(p$graph, "med>")))
  expect_true(any(startsWith(p$graph, "delta_w>")))
})

test_that("the native benchmark validates reps too", {
  dt <- geozl:::.datatype("uint8")
  bytes <- .Call(geozl:::geozl_r_pack, matrix(0:63, 8, 8), dt$code, NULL)
  bench <- function(reps) .Call(
    geozl:::geozl_r_bench, bytes, "id>zstd", 8, 1, NULL, dt$code,
    0L, NULL, reps, FALSE
  )
  expect_error(bench(Inf), "whole number")
  expect_error(bench(.Machine$integer.max + 1), "whole number")
})
