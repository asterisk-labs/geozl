test_that("every frame released by GeoZL still decodes", {
  dir <- golden_dir()
  skip_if(!nzchar(dir), "golden frames not found; set GEOZL_GOLDEN_DIR")
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("digest")
  manifest <- jsonlite::fromJSON(file.path(dir, "manifest.json"),
                                 simplifyVector = FALSE)
  expect_gt(length(manifest$frames), 0)
  for (entry in manifest$frames) {
    path <- file.path(dir, "frames", entry$file)
    frame <- readBin(path, "raw", file.size(path))
    expect_identical(digest::digest(frame, "sha256", serialize = FALSE),
                     entry$sha256_frame, label = entry$name)
    decoded <- geozl_decompress(frame)
    expect_identical(digest::digest(decoded, "sha256", serialize = FALSE),
                     entry$sha256_decoded, label = entry$name)
  }
})
