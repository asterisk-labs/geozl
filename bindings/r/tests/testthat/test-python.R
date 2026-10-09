# R and Python write the same frame for the same samples and settings. Runs
# when GEOZL_PYTHON names a Python with geozl, as `make r-test` sets it.
cases <- list(
  list(x = round(smooth_raster(64, 48, 0, 4000)), method = "planar>zigzag>pfor",
       datatype = "uint16"),
  list(x = array(round(smooth_raster(32, 24 * 3, 0, 900)), c(32, 24, 3)),
       method = "planar>zigzag>pivco", datatype = "uint16"),
  list(x = round(smooth_raster(40, 40, 0, 255)), method = "med>zigzag>entropy",
       datatype = "uint8"),
  list(x = smooth_raster(48, 40, 0, 500), method = "planar>zigzag>transpose>entropy",
       datatype = "float32", error = 0.5),
  list(x = smooth_raster(48, 40, 1, 500), method = "planar>zigzag>transpose>entropy",
       datatype = "float64", error = "1%"),
  list(x = round(smooth_raster(30, 30, -100, 100)), method = "planar>zigzag>pfor",
       datatype = "int16", nodata = -9999),
  list(x = smooth_raster(16, 16, -3, 3), method = "id>zstd", datatype = "float16"),
  list(x = smooth_raster(40, 30, 0, 1), method = "planar>zigzag>transpose>entropy",
       datatype = "float64", holes = TRUE)
)

test_that("R and Python write identical frames", {
  skip_if(!nzchar(Sys.getenv("GEOZL_PYTHON")), "GEOZL_PYTHON not set")
  for (case in cases) {
    x <- case$x
    if (!is.null(case$nodata))
      x[c(4, 9)] <- NA
    if (isTRUE(case$holes))
      x[c(2, 300, 400)] <- c(NaN, NA, NaN)
    g <- geozl_graph(x, case$method, error = case$error, nodata = case$nodata,
                     datatype = case$datatype)
    expect_identical(geozl_compress(x, g),
                     python_frame(x, case$method, case$datatype, case$error,
                                  case$nodata),
                     label = paste(case$method, case$datatype))
  }
})
