"""
    GeoZLError(code, message)

Raised when the GeoZL library refuses a tile or a frame. `code` is the ZL error
code as an `Int`, or `nothing` when the refusal has none.
"""
struct GeoZLError <: Exception
    code::Union{Nothing,Int}
    message::String
end

Base.showerror(io::IO, e::GeoZLError) = print(io, "GeoZLError: ", e.message)

# Codes 21 and 30 mean an invalid request here, not a corrupt frame.
const _INVALID_CODES = (21, 30)

_refusal(code::Integer, message::AbstractString) =
    code in _INVALID_CODES ? ArgumentError(message) : GeoZLError(code, message)

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

# Match Python's shortest round-tripping float spelling in recipe strings.
function _python_float(x::Union{Float16,Float32,Float64})
    text = replace(string(x), r"^Float16\((.*)\)$" => s"\1")
    m = match(r"^(\d+)(?:\.(\d+))?(?:[ef]([+-]?\d+))?$", text)
    digits = m[1] * something(m[2], "")
    point = length(m[1]) + (m[3] === nothing ? 0 : parse(Int, m[3]))
    lead = findfirst(!=('0'), digits)
    digits = rstrip(digits[lead:end], '0')
    point -= lead - 1
    limit = x isa Float16 ? 1e3 : x isa Float32 ? 1e6 : 1e16
    if 1e-4 <= Float64(x) < limit
        point <= 0 && return "0." * "0"^-point * digits
        point >= length(digits) && return digits * "0"^(point - length(digits)) * ".0"
        return digits[1:point] * "." * digits[point+1:end]
    end
    mantissa = length(digits) == 1 ? digits : digits[1:1] * "." * digits[2:end]
    return mantissa * "e" * (point > 0 ? "+" : "-") * lpad(abs(point - 1), 2, '0')
end

# Normalize shorthand so the three bindings write the same recipe.
function _normalize_error(error)
    error === nothing && return nothing
    error isa Bool && throw(ArgumentError("`error` must not be a Bool"))
    if error isa Real
        isfinite(error) || throw(ArgumentError("`error` must be finite, got $error"))
        error < 0 && throw(ArgumentError("`error` must not be negative, got $error"))
        iszero(error) && return nothing
        error isa Integer && return "LINEAR:MAX_ERROR=$error"
        value = error isa Union{Float16,Float32,Float64} ? error : Float64(error)
        isfinite(value) || throw(ArgumentError("`error` cannot be represented as a finite number"))
        iszero(value) && throw(ArgumentError("`error` $error is too small for a Float64"))
        return "LINEAR:MAX_ERROR=" * _python_float(value)
    end
    error isa AbstractString ||
        throw(ArgumentError("`error` must be a number, percentage or recipe, got $(repr(error))"))
    '\0' in error && throw(ArgumentError("`error` must not contain a NUL character"))
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
