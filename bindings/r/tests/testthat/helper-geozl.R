# Smooth test rasters: a gradient of sines spanning [lo, hi], shaped (x, y).
smooth_raster <- function(nx, ny, lo, hi) {
  v <- outer(seq_len(nx), seq_len(ny), function(i, j) sin(i / 9) + cos(j / 7))
  lo + (v - min(v)) / diff(range(v)) * (hi - lo)
}

# The golden frames frozen by the Python tests, found from the repository or
# from GEOZL_GOLDEN_DIR.
golden_dir <- function() {
  dir <- Sys.getenv("GEOZL_GOLDEN_DIR")
  if (!nzchar(dir))
    dir <- test_path("..", "..", "..", "python", "test", "golden")
  if (file.exists(file.path(dir, "manifest.json"))) normalizePath(dir) else ""
}

# The frame Python writes for the same samples and settings, or NULL when no
# Python with geozl is named by GEOZL_PYTHON.
python_frame <- function(x, method, datatype, error = NULL, nodata = NULL,
                         width = NULL, planes = NULL) {
  python <- Sys.getenv("GEOZL_PYTHON")
  if (!nzchar(python))
    return(NULL)
  dt <- geozl:::.datatype(datatype)
  samples <- tempfile(fileext = ".bin")
  frame <- tempfile(fileext = ".zl")
  on.exit(unlink(c(samples, frame)))
  writeBin(.Call(geozl:::geozl_r_pack, x, dt$code,
                 if (!is.null(nodata) && !is.na(nodata)) nodata), samples)
  shape <- if (is.null(dim(x))) length(x) else rev(dim(x))
  arg <- function(v) if (is.null(v)) "None" else if (is.character(v))
    sprintf("'%s'", v) else if (is.na(v)) "float('nan')" else format(v, digits = 17)
  script <- sprintf(paste(
    "import numpy as np, geozl",
    "a = np.fromfile('%s', '%s').reshape(%s)",
    "g = geozl.graph(a, '%s', error=%s, nodata=%s, width=%s, planes=%s)",
    "open('%s', 'wb').write(geozl.compress(a, graph=g))", sep = "\n"),
    samples, dt$name, paste0("(", paste(shape, collapse = ", "), ",)"), method,
    arg(error), arg(nodata), arg(width), arg(planes), frame)
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
    stop(paste(c("python could not write the frame:", output), collapse = "\n"),
         call. = FALSE)
  readBin(frame, "raw", file.size(frame))
}
