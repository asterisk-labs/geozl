const PRIORS = ("planar", "med", "delta_w", "delta_n", "average", "wp_static", "delta_1d", "none")

const ProfileRow = NamedTuple{(:graph, :bytes, :ratio, :encode_mbps, :decode_mbps, :shannon_pct),
                              Tuple{String,Int,Float64,Float64,Float64,Float64}}

"""
    Profile

The rows [`profile`](@ref) returns, best ratio first, with the input and the
settings that made them. It is a vector of named tuples with fields `graph`,
`bytes`, `ratio`, `encode_mbps`, `decode_mbps` and `shannon_pct`.
"""
struct Profile <: AbstractVector{ProfileRow}
    rows::Vector{ProfileRow}
    dims::Tuple
    eltype::DataType
    raw_bytes::Int
    width::Int
    planes::Int
    prior::Union{Nothing,String}
    reps::Int
    error::Union{Nothing,String}
end

Base.size(p::Profile) = size(p.rows)
Base.getindex(p::Profile, i::Int) = p.rows[i]

function Base.show(io::IO, ::MIME"text/plain", p::Profile)
    axes = get(Dict(1 => "samples", 2 => "columns, rows", 3 => "columns, rows, planes"),
               length(p.dims), "columns, rows, ..., planes")
    predictors = p.prior === nothing ? "all predictors" :
                 p.prior == "none" ? "no predictor" : "$(p.prior) + id"
    println(io, "input")
    println(io, "  size     ", p.dims, "  [", axes, "]")
    println(io, "  eltype   ", p.eltype, "  [", sizeof(p.eltype), " bytes/sample]")
    @printf(io, "  raw      %.2f MB\n", p.raw_bytes / 1e6)
    println(io, "  profile  ", p.planes, p.planes == 1 ? " plane" : " planes", ", row width ",
            p.width, ", ", predictors, ", ", p.reps, p.reps == 1 ? " rep" : " reps", ", ",
            something(p.error, "lossless"))
    println(io)
    if isempty(p.rows)
        print(io, "no compatible graphs")
        return
    end
    @printf(io, "%-32s %6s %9s %9s %6s", "graph", "ratio", "enc MB/s", "dec MB/s", "shan%")
    for r in p.rows
        @printf(io, "\n%-32s %6.2f %9.1f %9.1f %6.0f", r.graph, r.ratio, r.encode_mbps,
                r.decode_mbps, r.shannon_pct)
    end
end

_mbps(bytes, seconds) = seconds <= 0 ? Inf : bytes / seconds / 1e6

# Order-0 entropy of the bytes, in bits per byte.
function _order0_bits(a::Array)
    bytes = reinterpret(UInt8, vec(a))
    isempty(bytes) && return 0.0
    counts = zeros(Int, 256)
    for b in bytes
        counts[b+1] += 1
    end
    p = counts[counts .> 0] ./ length(bytes)
    return -sum(p .* log2.(p))
end

function _grid(prior, itemsize)
    stride, cap = 48, 128
    names = zeros(UInt8, stride * cap)
    count = Ref{Csize_t}(0)
    rc = ccall(_sym(:geozl_2d_grid_c), Cint,
               (Cstring, Csize_t, Ptr{UInt8}, Csize_t, Csize_t, Ref{Csize_t}),
               something(prior, ""), itemsize, names, stride, cap, count)
    rc == 0 || throw(ArgumentError("prior $(repr(prior)) is not one of $PRIORS or nothing"))
    return [_errmsg(names[(i-1)*stride+1:i*stride]) for i in 1:min(Int(count[]), cap)]
end

"""
    profile(x; prior="planar", width=nothing, planes=nothing, error=nothing,
            reps=5, nodata=nothing, verify=false) -> Profile

Compress and decompress `x` with every recipe `prior` allows and rank them by
ratio. `prior` is one predictor plus the identity path: `"planar"`, `"med"`,
`"delta_w"`, `"delta_n"`, `"average"`, `"wp_static"` or `"delta_1d"`;
`nothing` tries every predictor and `"none"` only the no-predictor path.
Recipes that do not apply to the input are left out. Other arguments match
[`graph`](@ref); `verify` checks checksums while timing decode.
"""
function profile(x::AbstractArray; prior::Union{Nothing,AbstractString}="planar",
                 width=nothing, planes=nothing, error=nothing, reps::Integer=5,
                 nodata=nothing, verify::Bool=false)
    prior !== nothing && isempty(prior) &&
        throw(ArgumentError("`prior` must be a non-empty string or nothing"))
    reps >= 1 || throw(ArgumentError("`reps` must be at least 1, got $reps"))
    recipe = _normalize_error(error)
    a = _samples(x, nodata)
    T = eltype(a)
    mode, bits = _nodata(T, a, nodata)
    w, p = _geometry(x, width, planes)
    raw = sizeof(a)
    ideal = raw * _order0_bits(a) / 8
    rows = ProfileRow[]
    for name in _grid(prior, sizeof(T))
        comp = Ref{Csize_t}(0)
        enc = Ref{Cdouble}(0)
        dec = Ref{Cdouble}(0)
        buf = zeros(UInt8, _ERR_SIZE)
        # Checksums on, so the size is the frame compress writes.
        GC.safepoint()
        rc = GC.@preserve a ccall(_sym(:geozl_2d_bench_c), Cint,
            (Cstring, UInt32, UInt32, Ptr{UInt8}, Cint, Cint, UInt64, Ptr{Cvoid}, Csize_t,
             Csize_t, Csize_t, Cint, Cint, Ref{Csize_t}, Ref{Cdouble}, Ref{Cdouble},
             Ptr{UInt8}, Csize_t),
            name, w, p, recipe === nothing ? C_NULL : recipe, _CODES[T], mode, bits, a,
            length(a), sizeof(T), reps, 1, verify, comp, enc, dec, buf, length(buf))
        GC.safepoint()
        rc == 0 || continue
        nbytes = Int(comp[])
        push!(rows, (graph = name, bytes = nbytes, ratio = raw / nbytes,
                     encode_mbps = _mbps(raw, enc[]), decode_mbps = _mbps(raw, dec[]),
                     shannon_pct = 100 * ideal / nbytes))
    end
    sort!(rows; by = r -> r.ratio, rev = true)
    return Profile(rows, size(x), T, raw, Int(w), Int(p),
                   prior === nothing ? nothing : String(prior), Int(reps), recipe)
end
