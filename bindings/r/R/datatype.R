# GeoZL's dtype codes, in the order of geozl/dtype.h.
.codes <- c(uint8 = 0L, uint16 = 1L, uint32 = 2L, uint64 = 3L,
            int8 = 4L, int16 = 5L, int32 = 6L, int64 = 7L,
            float16 = 8L, float32 = 9L, float64 = 10L)
.itemsizes <- c(1L, 2L, 4L, 8L, 1L, 2L, 4L, 8L, 2L, 4L, 8L)

# terra's datatype names and GDAL's type names for the same storage.
.aliases <- c(
  INT1U = "uint8", INT2U = "uint16", INT4U = "uint32", INT8U = "uint64",
  INT1S = "int8", INT2S = "int16", INT4S = "int32", INT8S = "int64",
  FLT4S = "float32", FLT8S = "float64",
  Byte = "uint8", UInt16 = "uint16", UInt32 = "uint32", UInt64 = "uint64",
  Int8 = "int8", Int16 = "int16", Int32 = "int32", Int64 = "int64",
  Float16 = "float16", Float32 = "float32", Float64 = "float64"
)

# The storage type of a raster: the datatype given, or the one R's own type
# implies.
.datatype <- function(datatype, x = NULL) {
  if (is.null(datatype)) {
    if (is.null(x))
      stop("`datatype` is required", call. = FALSE)
    datatype <- switch(typeof(x),
      raw = "uint8", integer = "int32", double = "float64",
      stop(sprintf("`x` must be a double, integer or raw vector, not %s",
                   typeof(x)), call. = FALSE))
  }
  if (!is.character(datatype) || length(datatype) != 1L || is.na(datatype))
    stop("`datatype` must be a single string", call. = FALSE)
  name <- if (datatype %in% names(.aliases)) .aliases[[datatype]] else datatype
  if (!name %in% names(.codes))
    stop(sprintf(paste0("unknown datatype \"%s\"; use one of %s, or terra's ",
                        "(\"INT2U\") or GDAL's (\"UInt16\") names"),
                 datatype, paste(names(.codes), collapse = ", ")),
         call. = FALSE)
  code <- .codes[[name]]
  list(name = name, code = code, itemsize = .itemsizes[[code + 1L]],
       integer = code <= .codes[["int64"]])
}
