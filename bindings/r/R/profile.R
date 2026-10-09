.priors <- c("planar", "med", "delta_w", "delta_n", "average", "wp_static",
             "delta_1d", "none")

.mbps <- function(bytes, seconds) if (seconds <= 0) Inf else bytes / seconds / 1e6

# Order-0 entropy of the bytes, in bits per byte.
.order0_bits <- function(bytes) {
  if (length(bytes) == 0L)
    return(0)
  p <- tabulate(as.integer(bytes) + 1L, 256L) / length(bytes)
  p <- p[p > 0]
  -sum(p * log2(p))
}

#' Benchmark candidate recipes on a tile
#'
#' Compresses and decompresses `x` with every recipe the prior allows and
#' ranks them by compression ratio. Recipes that do not apply to the input are
#' left out.
#'
#' @inheritParams geozl_graph
#' @param prior One predictor plus the identity path: `"planar"`, `"med"`,
#'   `"delta_w"`, `"delta_n"`, `"average"`, `"wp_static"` or `"delta_1d"`.
#'   `NULL` tries every predictor and `"none"` only the no-predictor path.
#' @param reps Round trips per recipe; the best time is kept.
#' @param verify Whether to check checksums while timing decode. The frames
#'   carry checksums either way.
#' @return A data frame with one row per recipe: `graph`, `bytes`, `ratio`,
#'   `encode_mbps`, `decode_mbps` and `shannon_pct`, the order-0 entropy bound
#'   as a percentage of the frame size.
#' @export
geozl_profile <- function(x, prior = "planar", width = NULL, planes = NULL,
                          error = NULL, reps = 5L, nodata = NULL,
                          datatype = NULL, verify = FALSE) {
  reps <- .whole(reps, "reps")
  if (reps > .Machine$integer.max)
    stop(sprintf("`reps` must be at most %s", format(.Machine$integer.max)),
         call. = FALSE)
  if (!isTRUE(verify) && !isFALSE(verify))
    stop("`verify` must be TRUE or FALSE", call. = FALSE)
  if (!is.null(prior) && (!is.character(prior) || length(prior) != 1L ||
                          is.na(prior) || !nzchar(prior)))
    stop("`prior` must be a non-empty string or NULL", call. = FALSE)
  recipe <- .normalize_error(error)
  dt <- .datatype(datatype, x)
  input <- .prepare(x, dt, nodata)
  geometry <- .geometry(x, input$n, width, planes)
  names <- .Call(geozl_r_grid, if (is.null(prior)) "" else prior, dt$itemsize)
  if (is.null(names))
    stop(sprintf("prior \"%s\" is not one of %s or NULL", prior,
                 paste(.priors, collapse = ", ")), call. = FALSE)

  raw_bytes <- length(input$bytes)
  ideal <- raw_bytes * .order0_bits(input$bytes) / 8
  rows <- lapply(names, function(name) {
    r <- .Call(geozl_r_bench, input$bytes, name, geometry$width,
               geometry$planes, recipe, dt$code, input$nodata$mode,
               input$nodata$value, reps, verify)
    if (is.null(r))
      return(NULL)
    data.frame(graph = name, bytes = r[[1]], ratio = raw_bytes / r[[1]],
               encode_mbps = .mbps(raw_bytes, r[[2]]),
               decode_mbps = .mbps(raw_bytes, r[[3]]),
               shannon_pct = 100 * ideal / r[[1]])
  })
  rows <- Filter(Negate(is.null), rows)
  out <- if (length(rows)) do.call(rbind, rows) else
    data.frame(graph = character(), bytes = numeric(), ratio = numeric(),
               encode_mbps = numeric(), decode_mbps = numeric(),
               shannon_pct = numeric())
  out <- out[order(-out$ratio), , drop = FALSE]
  rownames(out) <- NULL
  structure(out, dim_x = if (is.null(dim(x))) length(x) else dim(x),
            datatype = dt$name, itemsize = dt$itemsize, raw_bytes = raw_bytes,
            width = geometry$width, planes = geometry$planes, prior = prior,
            reps = reps, error = recipe,
            class = c("geozl_profile", "data.frame"))
}

#' @export
print.geozl_profile <- function(x, ...) {
  d <- attr(x, "dim_x")
  axes <- switch(as.character(length(d)), "1" = "samples",
                 "2" = "columns, rows", "3" = "columns, rows, planes",
                 "columns, rows, ..., planes")
  planes <- attr(x, "planes")
  reps <- attr(x, "reps")
  prior <- attr(x, "prior")
  predictors <- if (is.null(prior)) "all predictors" else
    if (prior == "none") "no predictor" else paste(prior, "+ id")
  mode <- if (is.null(attr(x, "error"))) "lossless" else attr(x, "error")
  cat("input\n")
  cat(sprintf("  dim      c(%s)  [%s]\n", paste(d, collapse = ", "), axes))
  cat(sprintf("  datatype %s  [%d bytes/sample]\n", attr(x, "datatype"),
              attr(x, "itemsize")))
  cat(sprintf("  raw      %.2f MB\n", attr(x, "raw_bytes") / 1e6))
  cat(sprintf("  profile  %s %s, row width %s, %s, %s %s, %s\n\n",
              format(planes), if (planes == 1) "plane" else "planes",
              format(attr(x, "width")), predictors, format(reps),
              if (reps == 1) "rep" else "reps", mode))
  if (nrow(x) == 0L) {
    cat("no compatible graphs\n")
    return(invisible(x))
  }
  cat(sprintf("%-32s %6s %9s %9s %6s\n", "graph", "ratio", "enc MB/s",
              "dec MB/s", "shan%"))
  for (i in seq_len(nrow(x)))
    cat(sprintf("%-32s %6.2f %9.1f %9.1f %6.0f\n", x$graph[[i]], x$ratio[[i]],
                x$encode_mbps[[i]], x$decode_mbps[[i]], x$shannon_pct[[i]]))
  invisible(x)
}
