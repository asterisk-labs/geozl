# R and Python build the same graphs, write the same frames, refuse the same
# requests and read the same values. Runs when GEOZL_PYTHON names a Python
# with geozl, as `make r` sets it.
skip_without_python <- function() {
  skip_if(!nzchar(Sys.getenv("GEOZL_PYTHON")), "GEOZL_PYTHON not set")
  skip_if_not_installed("jsonlite")
}

cases <- list(
  list(x = round(smooth_raster(64, 48, 0, 4000)), method = "planar>zigzag>pfor",
       dtype = "uint16"),
  list(x = array(round(smooth_raster(32, 24 * 3, 0, 900)), c(32, 24, 3)),
       method = "planar>zigzag>lanes", dtype = "uint16"),
  list(x = round(smooth_raster(40, 40, 0, 255)), method = "med>zigzag>entropy",
       dtype = "uint8"),
  list(x = smooth_raster(48, 40, 0, 500), method = "planar>zigzag>transpose>entropy",
       dtype = "float32", error = 0.5),
  list(x = smooth_raster(48, 40, 1, 500), method = "planar>zigzag>transpose>entropy",
       dtype = "float64", error = "1%"),
  list(x = round(smooth_raster(30, 30, -100, 100)), method = "planar>zigzag>pfor",
       dtype = "int16", nodata = -9999),
  list(x = smooth_raster(16, 16, -3, 3), method = "id>zstd", dtype = "float16"),
  list(x = smooth_raster(40, 30, 0, 1), method = "planar>zigzag>transpose>entropy",
       dtype = "float64", holes = TRUE),
  list(x = round(smooth_raster(64, 48, 0, 4000)), method = "planar>zigzag>pfor",
       dtype = "uint16", width = 32, planes = 2),
  list(x = array(round(smooth_raster(16, 12 * 3, 0, 900)), c(16, 12, 3)),
       method = "planar>zigzag>pfor", dtype = "uint16", width = 16 * 12,
       planes = 1),
  list(x = as.vector(round(smooth_raster(30, 20, 0, 255))),
       method = "med>zigzag>entropy", dtype = "uint8", width = 30),
  list(x = matrix(integer(), 0, 8), method = "id>zstd", dtype = "int16")
)

test_that("R and Python build the same graphs and write identical frames", {
  skip_without_python()
  for (case in cases) {
    x <- case$x
    if (!is.null(case$nodata))
      x[c(4, 9)] <- NA
    if (isTRUE(case$holes))
      x[c(2, 300, 400)] <- c(NaN, NA, NaN)
    label <- paste(case$method, case$dtype, format(case$width),
                   format(case$planes))
    g <- geozl_graph(x, case$method, width = case$width, planes = case$planes,
                     error = case$error, nodata = case$nodata,
                     dtype = case$dtype)
    py <- python_graph(x, case$method, case$dtype, case$error, case$nodata,
                       case$width, case$planes)
    expect_identical(geozl_compress(x, g), py$frame, label = label)
    f <- py$fields
    expect_identical(g$method, f$method, label = label)
    expect_equal(c(g$width, g$planes, g$itemsize),
                 c(f$width, f$planes, f$itemsize), label = label)
    expect_identical(g$dtype, f$dtype, label = label)
    expect_identical(g$error, f$error, label = label)
    expect_equal(g$nodata, f$nodata, label = label)
  }
})

test_that("the error recipe normalizes as in Python", {
  skip_without_python()
  x <- smooth_raster(48, 40, 1, 500)
  errors <- list(0.5, 2, 1 / 3, 1e-5, 0, "1%", "0.5%", "0%",
                 "LINEAR:MAX_ERROR=0.25")
  calls <- vapply(errors, function(e) sprintf(paste(
    "out.append(geozl.graph(a, 'planar>zigzag>transpose>entropy',",
    "error=%s).error)"), py_arg(e, float = TRUE)), "")
  py <- python_json(paste(c("import json, numpy as np, geozl",
                            py_samples(x, "float64"), "out = []", calls,
                            "print(json.dumps(out))"), collapse = "\n"))
  for (i in seq_along(errors)) {
    r <- geozl_graph(x, "planar>zigzag>transpose>entropy",
                     error = errors[[i]])$error
    # R writes 2 where Python writes 2.0 for a whole bound.
    if (!identical(r, py[[i]]))
      r <- paste0(r, ".0")
    expect_identical(r, py[[i]], label = format(errors[[i]]))
  }
})

test_that("R and Python refuse the same requests the same way", {
  skip_without_python()
  tile <- round(smooth_raster(32, 32, 0, 4000))
  x <- smooth_raster(48, 40, 0, 500)
  frame <- geozl_compress(tile, geozl_graph(tile, "planar>zigzag>pfor",
                                            dtype = "uint16"))
  flipped <- frame
  flipped[length(flipped) - 4] <- xor(flipped[length(flipped) - 4], as.raw(1))
  garbage <- as.raw(0:63)
  py <- python_json(paste(
    "import json, numpy as np, geozl", py_caught,
    py_samples(tile, "uint16", "t"), py_samples(x, "float32", "x"),
    py_samples(x * 1e6, "float32", "far"), py_bytes(frame, "frame"),
    py_bytes(flipped, "flipped"), py_bytes(garbage, "garbage"),
    "g = geozl.graph(x, 'planar>zigzag>transpose>entropy', error=0.5)",
    "print(json.dumps({",
    "    'recipe': caught(lambda: geozl.graph(t, 'planar>nothing')),",
    "    'domain': caught(lambda: geozl.compress(far, graph=g)),",
    "    'limit': caught(lambda: geozl.decompress(frame,",
    "                                             max_output_size=100)),",
    "    'flipped': caught(lambda: geozl.decompress(flipped)),",
    "    'garbage': caught(lambda: geozl.decompress(garbage))}))", sep = "\n"))
  g <- geozl_graph(x, "planar>zigzag>transpose>entropy", error = 0.5,
                   dtype = "float32")
  r <- list(
    recipe = caught(geozl_graph(tile, "planar>nothing", dtype = "uint16")),
    domain = caught(geozl_compress(x * 1e6, g)),
    limit = caught(geozl_decompress(frame, max_output_size = 100)),
    flipped = caught(geozl_decompress(flipped)),
    garbage = caught(geozl_decompress(garbage)))
  for (name in names(r)) {
    e <- r[[name]]
    p <- py[[name]]
    expect_null(e$call, label = name)
    # A plain R error is a ValueError, a geozl_invalid one also a GeoZLError,
    # and a geozl_error a GeoZLError alone.
    expect_identical(p$value_error, !e$geozl_error, label = name)
    expect_identical(p$geozl_error && !p$value_error, e$geozl_error,
                     label = name)
    expect_identical(p$geozl_error && p$value_error, e$invalid, label = name)
    code <- if (is.null(e$code)) message_code(e$message) else e$code
    expect_identical(code, if (is.null(p$code)) NA_real_ else
                       as.numeric(p$code), label = name)
    expect_identical(reason(e$message), reason(p$message), label = name)
  }
})

test_that("R and Python profile alike and fail alike", {
  skip_without_python()
  t <- outer(1:48, 1:32, function(x, y) 1000 + 3 * (x - 1) + 2 * (y - 1))
  py <- python_json(paste(
    "import json, numpy as np, geozl", py_caught, py_samples(t, "uint16"),
    "def rows(prior):",
    "    r = geozl.profile(a, prior=prior, reps=1)",
    "    return {'graph': [row['graph'] for row in r],",
    "            'bytes': [row['bytes'] for row in r], 'shape': list(r.shape),",
    "            'dtype': str(r.dtype), 'raw_bytes': r.raw_bytes,",
    "            'width': r.width, 'planes': r.planes, 'prior': r.prior,",
    "            'reps': r.reps, 'error': r.error}",
    "print(json.dumps({'planar': rows('planar'), 'all': rows(None),",
    "    'failure': caught(lambda: geozl.profile(a, error='NOPE:1',",
    "                                            reps=1))}))",
    sep = "\n"))
  for (prior in c("planar", "all")) {
    p <- geozl_profile(t, prior = if (prior == "planar") prior, reps = 1,
                       dtype = "uint16")
    f <- py[[prior]]
    expect_identical(p$graph, unlist(f$graph), label = prior)
    expect_equal(p$bytes, unlist(f$bytes), label = prior)
    expect_equal(attr(p, "dims"), rev(unlist(f$shape)), label = prior)
    expect_identical(attr(p, "dtype"), f$dtype, label = prior)
    expect_equal(c(attr(p, "raw_bytes"), attr(p, "width"), attr(p, "planes"),
                   attr(p, "reps")),
                 c(f$raw_bytes, f$width, f$planes, f$reps), label = prior)
    expect_identical(attr(p, "prior"), f$prior, label = prior)
    expect_identical(attr(p, "error"), f$error, label = prior)
  }
  e <- caught(geozl_profile(t, error = "NOPE:1", reps = 1, dtype = "uint16"))
  expect_null(e$call)
  expect_identical(e$geozl_error,
                   py$failure$geozl_error && !py$failure$value_error)
  expect_identical(e$invalid,
                   py$failure$geozl_error && py$failure$value_error)
  expect_identical(e$code, as.numeric(py$failure$code))
  expect_identical(reason(e$message), reason(py$failure$message))
})

test_that("R reads 64-bit extremes as Python does and warns on lost digits", {
  skip_without_python()
  values <- list(
    uint64 = c("0", "1", "9007199254740992", "9007199254740993",
               "9007199254740994", "9223372036854775808",
               "18446744073709551615"),
    int64 = c("-9223372036854775808", "-9223372036854775807",
              "-9007199254740993", "-1", "0", "9007199254740993",
              "9223372036854775807"))
  for (dtype in names(values)) {
    path <- tempfile(fileext = ".zl")
    py <- python_json(paste(
      "import json, numpy as np, geozl",
      sprintf("a = np.array([%s], '%s')",
              paste(values[[dtype]], collapse = ", "), dtype),
      "f = geozl.compress(a, graph=geozl.graph(a, 'id>zstd', width=a.size))",
      sprintf("open('%s', 'wb').write(f)", path),
      sprintf("v = [int(x) for x in geozl.decompress(f).view('%s')]", dtype),
      "d = np.array([float(x) for x in v], np.float64)",
      paste("print(json.dumps({'read': [str(x) for x in v],",
            "'doubles': d.tobytes().hex(),",
            "'lost': sum(int(float(x)) != x for x in v)}))"), sep = "\n"))
    expect_identical(unlist(py$read), values[[dtype]], label = dtype)
    # R cannot parse 19 digits exactly, so Python sends each double's bytes.
    hex <- py$doubles
    starts <- seq(1L, nchar(hex), by = 2L)
    doubles <- readBin(as.raw(strtoi(substring(hex, starts, starts + 1L), 16L)),
                       "double", nchar(hex) / 16L)
    frame <- readBin(path, "raw", file.size(path))
    expect_warning(back <- geozl_decompress(frame, dtype),
                   sprintf("^%d %s values lost precision", py$lost, dtype))
    expect_identical(back, doubles, label = dtype)
  }
})
