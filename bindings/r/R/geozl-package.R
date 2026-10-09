#' geozl: geospatial codecs for OpenZL
#'
#' Compress raster tiles with GeoZL's spatial predictors, NoData mask and
#' bounded-error quantizers, and decode the frames any GeoZL reader writes.
#' Build a graph once with [geozl_graph()], compress tiles with
#' [geozl_compress()], and read them back with [geozl_decompress()].
#' [geozl_profile()] ranks recipes on a sample.
#'
#' @keywords internal
#' @useDynLib geozl, .registration = TRUE
"_PACKAGE"
