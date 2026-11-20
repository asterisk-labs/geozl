test_that("profile ranks the recipes a prior allows", {
  x <- round(smooth_raster(64, 64, 0, 4000))
  p <- geozl_profile(x, dtype = "uint16", reps = 1)
  expect_s3_class(p, "geozl_profile")
  expect_named(p, c("graph", "bytes", "ratio", "encode_mbps", "decode_mbps",
                    "shannon_pct"))
  expect_gt(nrow(p), 5)
  expect_true(all(diff(p$ratio) <= 0))
  expect_true(all(startsWith(p$graph, "planar>") | startsWith(p$graph, "id>")))
  expect_true("planar>zigzag>pfor" %in% p$graph)
  best <- geozl_graph(x, p$graph[[1]], dtype = "uint16")
  expect_identical(length(geozl_compress(x, best)), as.integer(p$bytes[[1]]))
  expect_output(print(p), "dec MB/s")
  expect_identical(attr(p, "dims"), c(64L, 64L))
  expect_identical(attr(p, "dtype"), "uint16")
  expect_identical(attr(p, "raw_bytes"), 64L * 64L * 2L)
  expect_null(attr(p, "itemsize"))
  expect_error(geozl_profile(x, prior = "bogus", dtype = "uint16"),
               "not one of")
  expect_error(geozl_profile(x, prior = "", dtype = "uint16"),
               "non-empty")
  expect_error(geozl_profile(x, reps = 0), "whole number")
  expect_error(geozl_profile(x, reps = Inf), "whole number")
  expect_error(geozl_profile(x, reps = .Machine$integer.max + 1), "at most")
})

test_that("a prior of NULL tries every predictor", {
  x <- round(smooth_raster(32, 32, 0, 255))
  p <- geozl_profile(x, prior = NULL, dtype = "uint8", reps = 1)
  expect_true(any(startsWith(p$graph, "med>")))
  expect_true(any(startsWith(p$graph, "delta_w>")))
})

test_that("a prior of id is the no-predictor path, as Python reports it", {
  x <- round(smooth_raster(16, 16, 0, 255))
  p <- geozl_profile(x, prior = "id", dtype = "uint8", reps = 1)
  expect_identical(attr(p, "prior"), "none")
  expect_true(any(grepl("no predictor", capture.output(print(p)), fixed = TRUE)))
})

test_that("the native benchmark validates reps too", {
  dt <- geozl:::.dtype("uint8")
  bytes <- .Call(geozl:::geozl_r_pack, matrix(0:63, 8, 8), dt$code, NULL)
  bench <- function(reps) .Call(
    geozl:::geozl_r_bench, bytes, "id>zstd", 8, 1, NULL, dt$code,
    0L, NULL, reps, FALSE
  )
  expect_error(bench(Inf), "whole number")
  expect_error(bench(.Machine$integer.max + 1), "whole number")
})

test_that("profile prints its settings as Python does", {
  x <- round(smooth_raster(64, 64, 0, 4000))
  out <- capture.output(print(geozl_profile(x, dtype = "uint16", reps = 1)))
  # A locale without the middle dot escapes it.
  out <- gsub("<U+00B7>", "\u00b7", out, fixed = TRUE)
  expect_identical(out[1:5], c(
    "input", "  dim      c(64, 64)  [columns, rows]",
    "  dtype    uint16  [2 bytes/sample]", "  raw      0.01 MB",
    paste("  profile  1 plane \u00b7 row width 64 \u00b7 planar + id \u00b7",
          "1 rep \u00b7 lossless")))
})

test_that("a profile where every recipe fails stops with the first reason", {
  x <- round(smooth_raster(32, 32, 0, 4000))
  e <- caught(geozl_profile(x, error = "NOPE:1", dtype = "uint16", reps = 1))
  expect_false(e$geozl_error)
  expect_true(e$invalid)
  expect_null(e$call)
  expect_true(startsWith(e$message, "geozl_profile failed: error \"NOPE:1\""))
  expect_identical(message_code(e$message), 21)
  expect_identical(e$code, 21)
})
