.whole <- function(value, name) {
  if (!is.numeric(value) || length(value) != 1L || is.na(value) ||
      !is.finite(value) || value < 1 || value != floor(value))
    stop(sprintf("`%s` must be a whole number of at least 1", name), call. = FALSE)
  as.double(value)
}

# Row width and plane count. The fastest-varying dimension is the row, so an
# array shaped (x, y) or (x, y, band), as stars lays out a raster, holds the
# same samples in the same order as Python's (y, x) or (band, y, x).
.geometry <- function(x, n, width, planes) {
  d <- dim(x)
  if (!is.null(d) && prod(d) != n)
    d <- NULL
  if (is.null(width)) {
    if (length(d) < 2L)
      stop("give `width` for a vector without dim", call. = FALSE)
    width <- d[[1L]]
  }
  if (is.null(planes))
    planes <- if (length(d) >= 3L) d[[length(d)]] else 1L
  width <- .whole(width, "width")
  planes <- .whole(planes, "planes")
  if (planes > 1 && (n %% planes != 0 || (n %/% planes) %% width != 0))
    stop(sprintf("%s planes do not split %s samples into whole rows of %s",
                 format(planes), format(n), format(width)), call. = FALSE)
  list(width = width, planes = planes)
}

# nodata as the C API takes it: mode 0 none, 1 NaN, 2 a sentinel value.
.nodata <- function(nodata, dt) {
  if (is.null(nodata))
    return(list(mode = 0L, value = NULL))
  if (length(nodata) != 1L || !is.numeric(nodata) && !identical(nodata, NA))
    stop("`nodata` must be a single number or NA", call. = FALSE)
  if (is.na(nodata)) {
    if (dt$integer)
      stop(sprintf("an %s raster cannot use NaN as nodata", dt$name),
           call. = FALSE)
    return(list(mode = 1L, value = NULL))
  }
  list(mode = 2L, value = as.double(nodata))
}

# Native samples of x, and the nodata mode a NaN in a float raster selects.
.prepare <- function(x, dt, nodata) {
  nd <- .nodata(nodata, dt)
  bytes <- .Call(geozl_r_pack, x, dt$code, nd$value)
  if (is.null(nodata) && !dt$integer && .Call(geozl_r_has_nan, bytes, dt$code))
    nd$mode <- 1L
  list(bytes = bytes, nodata = nd, n = length(bytes) %/% dt$itemsize)
}

#' Build a reusable compression graph
#'
#' A graph fixes the recipe, the storage type and the predictor geometry for
#' every tile compressed with it.
#'
#' The fastest-varying dimension of `x` is the row: `width` defaults to
#' `dim(x)[1]` and `planes` to the last dimension of an array with three or
#' more. An array shaped `(x, y, band)`, as `stars` holds a raster, therefore
#' produces the same frames as Python does for `(band, y, x)`. The matrix from
#' `terra::values(r)` is shaped `(ncell, nlyr)`, so pass
#' `width = terra::ncol(r), planes = terra::nlyr(r)`. Row-based predictors
#' reset at each plane; use `width = nx * ny, planes = 1` to predict across
#' bands.
#'
#' Samples are converted to `datatype` before compression. Integer types refuse
#' values that are not whole or do not fit, and float types round to nearest.
#' A raw vector is taken as native samples of `datatype` already.
#'
#' @param x Raster samples: a double, integer or raw vector or array. For a
#'   bounded-error graph it fixes the domain later tiles are checked against,
#'   so use data spanning the product.
#' @param method A recipe such as `"planar>zigzag>pfor"` or
#'   `"planar>zigzag>transpose>entropy"`; see [geozl_profile()].
#' @param width,planes Row width in samples and number of stacked images.
#' @param error `NULL` or `0` for lossless, a positive number for an absolute
#'   bound (LINEAR), a percentage such as `"1%"` (LOG), or a full `LINEAR`,
#'   `LOG` or `SQRT` recipe.
#' @param nodata A sentinel value marking missing samples. `NA` in `x` is
#'   stored as this value. `NaN` selects NaN handling, which float rasters with
#'   missing values get without asking.
#' @param datatype Storage type: `"uint16"`, terra's `"INT2U"` or GDAL's
#'   `"UInt16"`, and so on for 8, 16, 32 and 64 bit integers and 16, 32 and 64
#'   bit floats. Defaults to `"int32"` for integer, `"float64"` for double and
#'   `"uint8"` for raw input.
#' @return A `geozl_graph`. It wraps native state, so it is not thread-safe and
#'   does not survive saving and reloading.
#' @examples
#' tile <- matrix(1000L + 1:(64 * 64) %% 97L, 64, 64)
#' g <- geozl_graph(tile, "planar>zigzag>pfor", datatype = "uint16")
#' frame <- geozl_compress(tile, g)
#' back <- geozl_decompress(frame, "uint16", dim = dim(tile), as = "integer")
#' identical(back, tile)
#' @export
geozl_graph <- function(x, method, width = NULL, planes = NULL, error = NULL,
                        nodata = NULL, datatype = NULL) {
  if (!is.character(method) || length(method) != 1L || is.na(method) ||
      !nzchar(method))
    stop("`method` must be a recipe such as \"planar>zigzag>pfor\"",
         call. = FALSE)
  recipe <- .normalize_error(error)
  dt <- .datatype(datatype, x)
  input <- .prepare(x, dt, nodata)
  geometry <- .geometry(x, input$n, width, planes)
  ptr <- .Call(geozl_r_graph_open, input$bytes, method, geometry$width,
               geometry$planes, recipe, dt$code, input$nodata$mode,
               input$nodata$value)
  structure(list(ptr = ptr, method = method, width = geometry$width,
                 planes = geometry$planes, datatype = dt$name, error = recipe,
                 nodata = nodata),
            class = "geozl_graph")
}

#' @export
print.geozl_graph <- function(x, ...) {
  plane_word <- if (x$planes == 1) "plane" else "planes"
  parts <- c(x$method, x$datatype, paste("row width", format(x$width)),
             paste(format(x$planes), plane_word),
             if (is.null(x$error)) "lossless" else x$error)
  cat("<geozl_graph> ", paste(parts, collapse = ", "), "\n", sep = "")
  invisible(x)
}

#' Compress a tile
#'
#' @param x Tile samples, converted to the graph's datatype. Its geometry is
#'   not checked against the graph.
#' @param graph A graph from [geozl_graph()].
#' @param coeffs Optional list of integer vectors to attach to this frame. Read
#'   them back with [geozl_coeffs()]; geozl assigns them no meaning.
#' @return The frame, as a raw vector.
#' @export
geozl_compress <- function(x, graph, coeffs = NULL) {
  if (!inherits(graph, "geozl_graph"))
    stop("`graph` must come from geozl_graph()", call. = FALSE)
  dt <- .datatype(graph$datatype)
  value <- if (!is.null(graph$nodata) && !is.na(graph$nodata))
    as.double(graph$nodata)
  bytes <- .Call(geozl_r_pack, x, dt$code, value)
  .Call(geozl_r_compress, graph$ptr, bytes, coeffs, dt$itemsize)
}

#' Decompress a frame
#'
#' A frame does not record its datatype or shape; keep them beside it, as Rumi
#' does.
#'
#' @param frame A frame, as a raw vector.
#' @param datatype The storage type the frame was written with. Without it the
#'   result is the raw native bytes.
#' @param dim Optional dimensions for the result.
#' @param as `"double"` (the default when `datatype` is given, as terra and
#'   stars hold rasters), `"integer"`, or `"raw"` for the native bytes, which
#'   torch reads without a copy through `torch_tensor_from_buffer()`.
#' @param nodata Samples equal to this value become `NA`.
#' @param verify Whether to check the frame's checksums.
#' @param max_output_size Refuse frames that declare more output bytes than
#'   this; set it when reading untrusted frames.
#' @return A vector of samples, with `dim` when given.
#' @export
geozl_decompress <- function(frame, datatype = NULL, dim = NULL, as = NULL,
                             nodata = NULL, verify = TRUE,
                             max_output_size = NULL) {
  if (!is.raw(frame))
    stop("`frame` must be a raw vector", call. = FALSE)
  if (!isTRUE(verify) && !isFALSE(verify))
    stop("`verify` must be TRUE or FALSE", call. = FALSE)
  if (!is.null(max_output_size) &&
      (!is.numeric(max_output_size) || length(max_output_size) != 1L ||
       is.na(max_output_size)))
    stop("`max_output_size` must be a single number", call. = FALSE)
  as <- if (is.null(as)) {
    if (is.null(datatype)) "raw" else "double"
  } else {
    match.arg(as, c("double", "integer", "raw"))
  }
  bytes <- .Call(geozl_r_decompress, frame, verify, max_output_size)
  if (is.null(datatype)) {
    if (as != "raw")
      stop("`datatype` is needed to return samples", call. = FALSE)
    out <- bytes
    n <- length(bytes)
  } else {
    dt <- .datatype(datatype)
    if (!is.null(dim) && as == "raw" && dt$itemsize > 1L)
      stop("`dim` counts samples, and as = \"raw\" returns bytes", call. = FALSE)
    value <- if (!is.null(nodata) && !is.na(nodata)) as.double(nodata)
    out <- .Call(geozl_r_unpack, bytes, dt$code,
                 c(raw = 0L, double = 1L, integer = 2L)[[as]], value)
    n <- length(bytes) %/% dt$itemsize
  }
  if (!is.null(dim)) {
    if (prod(dim) != n)
      stop(sprintf("`dim` holds %s samples, the frame %s", format(prod(dim)),
                   format(n)), call. = FALSE)
    dim(out) <- dim
  }
  out
}

#' Read the coefficients attached to a frame
#'
#' Only the frame header is read; the payload is not decompressed.
#'
#' @param frame A frame, as a raw vector.
#' @return A list of integer vectors, or `NULL` when the frame carries none.
#' @export
geozl_coeffs <- function(frame) {
  if (!is.raw(frame))
    stop("`frame` must be a raw vector", call. = FALSE)
  .Call(geozl_r_coeffs, frame)
}
