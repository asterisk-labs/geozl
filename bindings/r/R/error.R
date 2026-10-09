.number_re <- "^[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?$"

# The shortest form that reads back as x. Whole numbers print as integers, the
# way Python writes an int bound, so both languages build the same recipe.
.format_number <- function(x) {
  if (x == floor(x) && abs(x) < 1e15)
    return(format(x, scientific = FALSE))
  for (digits in 15:17) {
    text <- sprintf("%.*g", digits, x)
    if (as.numeric(text) == x)
      break
  }
  text
}

# Whether a simple or canonical recipe declares zero error.
.zero_recipe <- function(error) {
  forms <- list(c("LINEAR:MAX_ERROR=", ""), c("LOG:MAX_ERROR=", "%"),
                c("SQRT:MAX_ERROR=", "N"))
  for (form in forms) {
    if (!startsWith(error, form[[1]]) || !endsWith(error, form[[2]]))
      next
    value <- substr(error, nchar(form[[1]]) + 1L, nchar(error) - nchar(form[[2]]))
    if (grepl(.number_re, value, perl = TRUE))
      return(as.numeric(value) == 0)
  }
  FALSE
}

# The short error syntax expanded to a codec recipe, NULL for lossless: a
# number is LINEAR, "1%" is LOG, and a full recipe passes through.
.normalize_error <- function(error) {
  if (is.null(error))
    return(NULL)
  if (is.logical(error))
    stop("`error` must not be a logical", call. = FALSE)
  if (length(error) != 1L || is.na(error))
    stop("`error` must be a single number, percentage or recipe", call. = FALSE)
  if (is.numeric(error)) {
    if (!is.finite(error))
      stop(sprintf("`error` must be finite, got %s", format(error)), call. = FALSE)
    if (error < 0)
      stop(sprintf("`error` must not be negative, got %s", format(error)),
           call. = FALSE)
    if (error == 0)
      return(NULL)
    return(paste0("LINEAR:MAX_ERROR=", .format_number(error)))
  }
  if (!is.character(error))
    stop("`error` must be a number, percentage or recipe", call. = FALSE)
  if (.zero_recipe(error))
    return(NULL)
  if (grepl(":", error, fixed = TRUE))
    return(error)
  if (endsWith(error, "%")) {
    value <- substr(error, 1L, nchar(error) - 1L)
    if (!grepl(.number_re, value, perl = TRUE))
      stop(sprintf("relative error must look like \"10%%\", got \"%s\"", error),
           call. = FALSE)
    amount <- as.numeric(value)
    if (amount == 0)
      return(NULL)
    if (amount < 0 || amount >= 100)
      stop(sprintf("relative error must be between 0%% and 100%%, got \"%s\"",
                   error), call. = FALSE)
    return(paste0("LOG:MAX_ERROR=", value, "%"))
  }
  stop(sprintf(paste0("`error` must be a number, a percentage such as \"10%%\", ",
                      "or a LINEAR, LOG or SQRT recipe, got \"%s\""), error),
       call. = FALSE)
}
