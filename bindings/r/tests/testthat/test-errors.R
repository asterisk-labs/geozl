tile <- round(smooth_raster(32, 32, 0, 4000))

test_that("values that do not fit the dtype are refused", {
  expect_error(geozl_graph(tile + 70000, "id>zstd", dtype = "uint16"),
               "does not fit uint16")
  expect_error(geozl_graph(tile + 0.5, "id>zstd", dtype = "int32"),
               "does not fit int32")
  expect_error(geozl_graph(tile * 1e40, "id>zstd", dtype = "float32"),
               "float32 range")
  expect_error(geozl_graph(tile * 100, "id>zstd", dtype = "float16"),
               "float16 range")
  expect_error(geozl_graph(tile, "id>zstd", dtype = "uint12"),
               "unknown dtype")
  expect_error(geozl_graph(matrix(TRUE, 8, 8), "id>zstd"), "logical")
})

test_that("NA in an integer raster needs nodata", {
  x <- tile
  x[3] <- NA
  expect_error(geozl_graph(x, "id>zstd", dtype = "uint16"),
               "needs `nodata`")
  expect_error(geozl_graph(tile, "id>zstd", nodata = NaN, dtype = "uint16"),
               "a uint16 raster cannot use NaN as nodata", fixed = TRUE)
  expect_error(geozl_graph(tile, "id>zstd", nodata = 70000, dtype = "uint16"),
               "nodata 70000 does not fit")
  expect_error(geozl_graph(tile, "id>zstd", nodata = c(1, 2)), "single number")
})

test_that("bad recipes and graphs fail with a reason", {
  e <- caught(geozl_graph(tile, "planar>nothing", dtype = "uint16"))
  expect_false(e$geozl_error)
  expect_true(e$invalid)
  expect_true(startsWith(e$message,
                         'geozl_graph failed (method = "planar>nothing"): '))
  expect_identical(message_code(e$message), 21)
  expect_identical(e$code, 21)
  expect_error(geozl_graph(tile, "", dtype = "uint16"), "recipe")
  expect_error(geozl_compress(tile, list()), "geozl_graph\\(\\)")
  g <- geozl_graph(tile, "id>zstd", dtype = "uint16")
  restored <- unserialize(serialize(g, NULL))
  expect_error(geozl_compress(tile, restored), "no longer open")
})

test_that("an invalid request keeps the plain error classes", {
  bad <- function() geozl_graph(tile, "planar>nothing", dtype = "uint16")
  e <- tryCatch(bad(), error = identity)
  expect_s3_class(e, c("geozl_invalid", "simpleError", "error", "condition"),
                  exact = TRUE)
  expect_identical(tryCatch(bad(), simpleError = function(e) "plain"), "plain")
  expect_error(tryCatch(bad(), geozl_error = function(e) "frame"),
               class = "geozl_invalid")
})

test_that("compress refuses a graph whose dtype field was edited", {
  g <- geozl_graph(tile, "id>zstd", dtype = "uint16")
  g$dtype <- "uint8"
  e <- caught(geozl_compress(matrix(7L, 32, 32), g))
  expect_identical(e$message, paste(
    "this graph was built for uint16 samples, but its `dtype` field names",
    "uint8; build it again with geozl_graph()"))
  expect_false(e$geozl_error)
  expect_null(e$call)
  # A wider field would have C read fewer samples than were packed.
  narrow <- geozl_graph(as.raw(1:64), "id>zstd", width = 64)
  narrow$dtype <- "float64"
  expect_error(geozl_compress(rep(1, 64), narrow),
               "built for uint8 samples, but its `dtype` field names float64",
               fixed = TRUE)
  # The dtype travels with the native graph, not with the list around it.
  g <- geozl_graph(tile, "id>zstd", dtype = "uint16")
  g$ptr <- narrow$ptr
  expect_error(geozl_compress(tile, g), "built for uint8 samples")
  g <- geozl_graph(tile, "id>zstd", dtype = "uint16")
  expect_error(.Call(geozl:::geozl_r_compress, g$ptr, as.raw(1:3), NULL, 1L,
                     g$method),
               "3 bytes do not hold whole uint16 samples", fixed = TRUE)
  expect_length(geozl_decompress(geozl_compress(tile, g)), 2 * length(tile))
})

# The varint of v, a whole number a double holds exactly. %% would warn past
# 2^52, so each group is taken by division alone.
varint <- function(v) {
  out <- raw()
  repeat {
    q <- floor(v / 128)
    out <- c(out, as.raw(v - q * 128 + if (q > 0) 128 else 0))
    if (q == 0)
      return(out)
    v <- q
  }
}

# A frame of 300 uint8 samples whose header stores `stored`, the declared size
# plus one, in place of 301. The header checksum after it is found by trying
# every byte until the size reads back.
declaring <- function(stored) {
  x <- rep(1L, 300)
  frame <- geozl_compress(x, geozl_graph(x, "id>zstd", width = 300,
                                         dtype = "uint8"))
  # Bytes 7 and 8 are the varint of 301, and byte 9 the header checksum.
  stopifnot(identical(frame[7:8], varint(301)))
  for (check in 0:255) {
    f <- c(frame[1:6], varint(stored), as.raw(check), frame[10:length(frame)])
    e <- caught(geozl_decompress(f, max_output_size = 0))
    if (grepl("declares", e$message))
      return(f)
  }
  stop("no header checksum fits")
}

test_that("a frame declaring more than an R vector holds is a frame error", {
  # 2^53 - 1 bytes passes R_XLEN_T_MAX, and 2^63 + 2047 turns negative as
  # R_xlen_t.
  for (stored in c(2^53, 2^63 + 2^11)) {
    f <- declaring(stored)
    for (args in list(list(), list(verify = FALSE), list(dtype = "uint8"))) {
      e <- caught(do.call(geozl_decompress, c(list(f), args)))
      expect_true(e$geozl_error)
      expect_match(e$message, paste("^geozl_decompress: the frame declares",
                                    "[0-9]+ bytes of output, more than an R",
                                    "vector holds$"))
      expect_identical(e$code, NA_real_)
      expect_null(e$call)
    }
    expect_error(geozl_decompress(f, max_output_size = 1e6),
                 "above the 1000000 allowed")
  }
})

test_that("damaged and oversized frames are refused", {
  g <- geozl_graph(tile, "planar>zigzag>pfor", dtype = "uint16")
  frame <- geozl_compress(tile, g)
  expect_error(geozl_decompress(frame[seq_len(length(frame) %/% 2)]),
               class = "geozl_error")
  expect_error(geozl_decompress(as.raw(0:63)), "unreadable frame",
               class = "geozl_error")
  expect_error(geozl_decompress(as.integer(frame)), "raw vector")
  e <- caught(geozl_decompress(frame, max_output_size = 100))
  expect_false(e$geozl_error)
  expect_identical(e$message, paste("geozl_decompress: the frame declares 2048",
                                    "bytes of output, above the 100 allowed"))
  flipped <- frame
  flipped[length(flipped) - 4] <- xor(flipped[length(flipped) - 4], as.raw(1))
  e <- caught(geozl_decompress(flipped))
  expect_true(e$geozl_error)
  expect_identical(e$code, message_code(e$message))
  expect_null(e$call)
  expect_error(geozl_decompress(frame, verify = NA), "TRUE or FALSE")
  expect_error(geozl_decompress(frame, as = "int"), "`as` must be")
  for (dim in list(c(32, NA), Inf, c(32, -Inf), -1, 2.5, "32", integer(),
                   c(2^31, 0), c(2^30, 2^30)))
    expect_error(geozl_decompress(frame, "uint16", dim = dim),
                 paste("`dim` must be whole numbers from 0 to 2147483647",
                       "with a product of at most 2^52"), fixed = TRUE)
  e <- caught(geozl_decompress(frame, "uint16", dim = Inf))
  expect_false(e$geozl_error)
  expect_null(e$call)
  expect_identical(dim(geozl_decompress(frame, "uint16", dim = c(32, 32))),
                   c(32L, 32L))
})

test_that("a frame that decodes to an invalid graph is a geozl_error", {
  frame <- geozl_compress(tile, geozl_graph(tile, "planar>zigzag>pfor",
                                            dtype = "uint16"))
  # Flip one bit at a time until decoding reports ZL error code 30.
  graph_invalid <- function() {
    for (i in seq_along(frame)) for (bit in 0:7) {
      f <- frame
      f[i] <- xor(f[i], as.raw(2^bit))
      e <- caught(geozl_decompress(f, verify = FALSE))
      if (!is.null(e) && message_code(e$message) %in% 30)
        return(e)
    }
    NULL
  }
  e <- graph_invalid()
  expect_false(is.null(e))
  expect_true(e$geozl_error)
  expect_false(e$invalid)
  expect_identical(e$code, 30)
  expect_null(e$call)
})

test_that("max_output_size must be a finite, non-negative number", {
  g <- geozl_graph(tile, "planar>zigzag>pfor", dtype = "uint16")
  frame <- geozl_compress(tile, g)
  for (limit in list(-1, Inf, NaN, NA, "100", c(1, 2), TRUE))
    expect_error(geozl_decompress(frame, max_output_size = limit),
                 "`max_output_size` must be a finite, non-negative number",
                 fixed = TRUE)
  expect_length(geozl_decompress(frame, max_output_size = 2048), 2048)
})

test_that("decompress checks nodata as geozl_graph does", {
  g <- geozl_graph(tile, "planar>zigzag>pfor", dtype = "uint16")
  frame <- geozl_compress(tile, g)
  for (nodata in list(NaN, NA))
    expect_error(geozl_decompress(frame, "uint16", nodata = nodata),
                 "a uint16 raster cannot use NaN as nodata", fixed = TRUE)
  expect_error(geozl_decompress(frame, "uint16", nodata = "1"),
               "single number")
  expect_error(geozl_decompress(frame, "uint16", nodata = c(1, 2)),
               "single number")
  expect_error(geozl_decompress(frame, "uint16", nodata = 70000),
               "nodata 70000 does not fit")
  expect_error(geozl_decompress(frame, "float16", nodata = -1),
               "a half float carries no sentinel", fixed = TRUE)
  back <- geozl_decompress(frame, "uint16", dim = dim(tile), nodata = tile[[1]])
  expect_identical(which(is.na(back)), which(tile == tile[[1]]))
})

test_that("nodata needs a dtype and samples to mark", {
  frame <- geozl_compress(tile, geozl_graph(tile, "id>zstd", dtype = "uint16"))
  e <- caught(geozl_decompress(frame, nodata = tile[[1]]))
  expect_identical(e$message, "`dtype` is needed to find `nodata` samples")
  expect_false(e$geozl_error)
  expect_null(e$call)
  e <- caught(geozl_decompress(frame, "uint16", as = "raw", nodata = tile[[1]]))
  expect_identical(e$message,
                   "`nodata` marks samples as NA, and as = \"raw\" returns bytes")
  expect_null(e$call)
  expect_identical(geozl_decompress(frame, as = "raw"), geozl_decompress(frame))
})

test_that("bad arguments are plain errors without a call", {
  g <- geozl_graph(tile, "planar>zigzag>pfor", dtype = "uint16")
  frame <- geozl_compress(tile, g)
  dt <- geozl:::.dtype("uint16")
  bytes <- .Call(geozl:::geozl_r_pack, tile, dt$code, NULL)
  errors <- list(
    caught(geozl_graph(tile, "id>zstd", width = 3.7)),
    caught(geozl_graph(tile + 70000, "id>zstd", dtype = "uint16")),
    caught(geozl_graph(tile, "id>zstd", nodata = 70000, dtype = "uint16")),
    caught(geozl_compress(tile, g, coeffs = list(1.5))),
    caught(geozl_decompress(frame, "uint16", as = "raw", dim = 4)),
    caught(.Call(geozl:::geozl_r_bench, bytes, "id>zstd", 0, 1, NULL, dt$code,
                 0L, NULL, 1, FALSE)),
    caught(.Call(geozl:::geozl_r_bench, bytes, "id>zstd", 8, 0, NULL, dt$code,
                 0L, NULL, 1, FALSE)))
  for (e in errors) {
    expect_false(e$geozl_error, label = e$message)
    expect_null(e$call, label = e$message)
  }
})

test_that("a refused lossy tile names the method", {
  x <- smooth_raster(48, 40, 0, 500)
  g <- geozl_graph(x, "planar>zigzag>transpose>entropy", error = 0.5,
                   dtype = "float32")
  e <- caught(geozl_compress(x * 1e6, g))
  expect_false(e$geozl_error)
  expect_true(e$invalid)
  expect_identical(e$code, 21)
  expect_true(startsWith(e$message, paste(
    'geozl_compress failed (method =',
    '"planar>zigzag>transpose>entropy"): ')))
  expect_null(e$call)
})

test_that("integer output needs values R can hold", {
  x <- matrix(4294967295, 4, 4)
  frame <- geozl_compress(x, geozl_graph(x, "id>zstd", dtype = "uint32"))
  expect_error(geozl_decompress(frame, "uint32", as = "integer"),
               "does not fit an R integer")
  expect_identical(geozl_decompress(frame, "uint32"), rep(4294967295, 16))
  f32 <- geozl_compress(tile, geozl_graph(tile, "id>zstd", dtype = "float32"))
  expect_error(geozl_decompress(f32, "float32", as = "integer"),
               "as = \"double\"")
  expect_error(geozl_decompress(f32, as = "double"), "`dtype` is needed")
})

test_that("64-bit values warn exactly when they lose digits as double", {
  frame_of <- function(bytes, dtype)
    geozl_compress(bytes, geozl_graph(bytes, "id>zstd",
                                      width = length(bytes) / 8, dtype = dtype))
  # 2^53 + 1 rounds to 2^53, while 2^53 + 2 and 2^60 are exact.
  expect_warning(back <- geozl_decompress(frame_of(le64(2^21, 1), "uint64"),
                                          "uint64"),
                 "1 uint64 values lost precision as double")
  expect_identical(back, 2^53)
  expect_no_warning(back <- geozl_decompress(frame_of(le64(2^21, 2), "uint64"),
                                             "uint64"))
  expect_identical(back, 2^53 + 2)
  expect_no_warning(back <- geozl_decompress(frame_of(le64(2^28, 0), "uint64"),
                                             "uint64"))
  expect_identical(back, 2^60)
  # 2^64 - 1 and 2^63 - 1 round up past their type.
  expect_warning(geozl_decompress(frame_of(le64(2^32 - 1, 2^32 - 1), "uint64"),
                                  "uint64"), "1 uint64")
  extremes <- c(le64(2^31, 0), le64(2^31 - 1, 2^32 - 1),
                le64(2^32 - 2^21 - 1, 2^32 - 1))
  expect_warning(back <- geozl_decompress(frame_of(extremes, "int64"), "int64"),
                 "2 int64 values lost precision as double")
  expect_identical(back, c(-2^63, 2^63, -2^53))
  # A sample read as NA loses nothing.
  sentinel <- frame_of(c(le64(2^21, 1), le64(0, 7)), "uint64")
  expect_warning(back <- geozl_decompress(sentinel, "uint64", nodata = 7),
                 "1 uint64")
  expect_identical(back, c(2^53, NA))
  expect_no_warning(geozl_decompress(frame_of(le64(0, 7), "uint64"), "uint64",
                                     nodata = 7))
})
