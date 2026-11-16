# Smooth test rasters: a gradient of sines spanning [lo, hi], shaped (x, y).
smooth_raster <- function(nx, ny, lo, hi) {
  v <- outer(seq_len(nx), seq_len(ny), function(i, j) sin(i / 9) + cos(j / 7))
  lo + (v - min(v)) / diff(range(v)) * (hi - lo)
}

# Eight little-endian bytes of the 64-bit integer hi * 2^32 + lo, with hi and
# lo whole numbers below 2^32.
le64 <- function(hi, lo) {
  word <- function(v) as.raw(v %/% 256^(0:3) %% 256)
  c(word(lo), word(hi))
}

# The golden corpus the Python tests keep, found from the repository or
# from GEOZL_GOLDEN_DIR.
golden_dir <- function() {
  dir <- Sys.getenv("GEOZL_GOLDEN_DIR")
  if (!nzchar(dir))
    dir <- test_path("..", "..", "..", "python", "test", "golden")
  if (file.exists(file.path(dir, "manifest.json"))) normalizePath(dir) else ""
}

# The output of a Python script, or NULL when no Python with geozl is named by
# GEOZL_PYTHON.
python_run <- function(script) {
  python <- Sys.getenv("GEOZL_PYTHON")
  if (!nzchar(python))
    return(NULL)
  env <- character()
  if (identical(Sys.info()[["sysname"]], "Linux")) {
    python_lib <- file.path(dirname(dirname(normalizePath(python))), "lib")
    library_path <- c(python_lib, Sys.getenv("LD_LIBRARY_PATH"))
    env <- paste0("LD_LIBRARY_PATH=", paste(library_path[nzchar(library_path)],
                                             collapse = .Platform$path.sep))
  }
  output <- suppressWarnings(system2(python, c("-c", shQuote(script)),
                                     env = env,
                                     stdout = TRUE, stderr = TRUE))
  status <- attr(output, "status")
  if (is.null(status))
    status <- 0L
  if (status != 0)
    stop(paste(c("python failed:", output), collapse = "\n"), call. = FALSE)
  output
}

# The last line a Python script prints, read as JSON.
python_json <- function(script) {
  output <- python_run(script)
  jsonlite::fromJSON(output[[length(output)]], simplifyVector = FALSE)
}

# Python text for an R value. With `float`, a number goes as a Python float.
py_arg <- function(v, float = FALSE) {
  if (is.null(v)) "None" else if (is.character(v)) sprintf("'%s'", v) else
    if (is.na(v)) "float('nan')" else if (float)
      sprintf("float('%s')", format(v, digits = 17)) else format(v, digits = 17)
}

# Write native samples of x to a file and return the Python that loads them
# as `name`, shaped as Python holds the same samples.
py_samples <- function(x, dtype, name = "a", nodata = NULL) {
  dt <- geozl:::.dtype(dtype)
  path <- tempfile(fileext = ".bin")
  writeBin(.Call(geozl:::geozl_r_pack, x, dt$code,
                 if (!is.null(nodata) && !is.na(nodata)) nodata), path)
  shape <- if (is.null(dim(x))) length(x) else rev(dim(x))
  sprintf("%s = np.fromfile('%s', '%s').reshape(%s)", name, path, dt$name,
          paste0("(", paste(shape, collapse = ", "), ",)"))
}

# Write a raw vector to a file and return the Python that reads it as `name`.
py_bytes <- function(bytes, name) {
  path <- tempfile(fileext = ".zl")
  writeBin(bytes, path)
  sprintf("%s = open('%s', 'rb').read()", name, path)
}

# Python's catch of an error, as JSON fields the R side compares.
py_caught <- paste(
  "def caught(f):",
  "    try:",
  "        f()",
  "    except Exception as e:",
  "        return {'value_error': isinstance(e, ValueError),",
  "                'geozl_error': isinstance(e, geozl.GeoZLError),",
  "                'code': getattr(e, 'code', None), 'message': str(e)}",
  "    return None", sep = "\n")

# The graph Python builds for the same samples and settings, its fields and
# the frame it writes for them, or NULL without GEOZL_PYTHON.
python_graph <- function(x, method, dtype, error = NULL, nodata = NULL,
                         width = NULL, planes = NULL) {
  if (!nzchar(Sys.getenv("GEOZL_PYTHON")))
    return(NULL)
  frame <- tempfile(fileext = ".zl")
  on.exit(unlink(frame))
  fields <- python_json(paste(
    "import json, numpy as np, geozl",
    py_samples(x, dtype, nodata = nodata),
    sprintf(paste("g = geozl.graph(a, '%s', error=%s, nodata=%s, width=%s,",
                  "planes=%s)"), method, py_arg(error), py_arg(nodata),
            py_arg(width), py_arg(planes)),
    sprintf("open('%s', 'wb').write(geozl.compress(a, graph=g))", frame),
    paste("print(json.dumps({'method': g.method, 'width': g.width,",
          "'planes': g.planes, 'dtype': str(g.dtype), 'itemsize': g.itemsize,",
          "'error': g.error, 'nodata': g.nodata}))"),
    sep = "\n"))
  list(fields = fields, frame = readBin(frame, "raw", file.size(frame)))
}

# An R error as the fields py_caught writes.
caught <- function(expr) {
  e <- tryCatch({
    expr
    NULL
  }, error = identity)
  if (is.null(e))
    return(NULL)
  list(geozl_error = inherits(e, "geozl_error"), code = e$code,
       message = conditionMessage(e), call = conditionCall(e))
}

# A message without the call prefix, `geozl_graph failed (method = "m"): ` in
# R and `geozl.graph failed (method='m'): ` in Python.
reason <- function(message)
  sub("^geozl[._][a-z]+( failed)?( \\(method ?= ?[^)]*\\))?: ", "", message)

# The ZL error code an error message ends with, or NA.
message_code <- function(message) {
  pattern <- "\\(ZL error code ([0-9]+)\\)$"
  m <- regmatches(message, regexec(pattern, message))[[1]]
  if (length(m)) as.numeric(m[[2]]) else NA_real_
}
