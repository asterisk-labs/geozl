#' geozl: Geospatial Codec Extensions for 'OpenZL'
#'
#' R bindings to libgeozl, the geospatial codec extensions for OpenZL.
#' Build a graph once with [geozl_graph()], compress tiles with
#' [geozl_compress()], and read them back with [geozl_decompress()].
#' [geozl_profile()] ranks recipes on a sample.
#'
#' @author Cesar Aybar \email{cesar@asterisk.coop}
#' @keywords internal
#' @useDynLib geozl, .registration = TRUE
"_PACKAGE"
