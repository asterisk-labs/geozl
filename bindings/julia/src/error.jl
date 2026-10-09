"""
    GeoZLError(msg)

Raised when the GeoZL library refuses a recipe, a tile or a frame.
"""
struct GeoZLError <: Exception
    msg::String
end

Base.showerror(io::IO, e::GeoZLError) = print(io, "GeoZLError: ", e.msg)

const _NUMBER = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"

# Whether a simple or canonical recipe declares zero error.
function _zero_recipe(error::AbstractString)
    isascii(error) || return false
    for (prefix, suffix) in (("LINEAR:MAX_ERROR=", ""), ("LOG:MAX_ERROR=", "%"),
                             ("SQRT:MAX_ERROR=", "N"))
        startswith(error, prefix) && endswith(error, suffix) || continue
        value = error[length(prefix)+1:end-length(suffix)]
        occursin(_NUMBER, value) && return iszero(parse(Float64, value))
    end
    return false
end

# The short error syntax expanded to a codec recipe, nothing for lossless: a
# number is LINEAR, "1%" is LOG, and a full recipe passes through. Numbers print
# as Python writes them, so both build the same recipe.
function _normalize_error(error)
    error === nothing && return nothing
    error isa Bool && throw(ArgumentError("`error` must not be a Bool"))
    if error isa Real
        isfinite(error) || throw(ArgumentError("`error` must be finite, got $error"))
        error < 0 && throw(ArgumentError("`error` must not be negative, got $error"))
        iszero(error) && return nothing
        value = error isa Integer || error isa AbstractFloat ? error : Float64(error)
        isfinite(value) || throw(ArgumentError("`error` cannot be represented as a finite number"))
        return "LINEAR:MAX_ERROR=$value"
    end
    error isa AbstractString ||
        throw(ArgumentError("`error` must be a number, percentage or recipe, got $(repr(error))"))
    _zero_recipe(error) && return nothing
    occursin(':', error) && return String(error)
    if endswith(error, "%")
        value = chop(error)
        occursin(_NUMBER, value) ||
            throw(ArgumentError("relative error must look like \"10%\", got $(repr(error))"))
        amount = parse(Float64, value)
        iszero(amount) && return nothing
        0 < amount < 100 ||
            throw(ArgumentError("relative error must be between 0% and 100%, got $(repr(error))"))
        return "LOG:MAX_ERROR=$value%"
    end
    throw(ArgumentError("`error` must be a number, a percentage such as \"10%\", " *
                        "or a LINEAR, LOG or SQRT recipe, got $(repr(error))"))
end
