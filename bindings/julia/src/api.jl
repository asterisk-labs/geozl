# GeoZL's dtype codes, in the order of geozl/dtype.h.
const _CODES = Dict{DataType,Cint}(
    UInt8 => 0, UInt16 => 1, UInt32 => 2, UInt64 => 3,
    Int8 => 4, Int16 => 5, Int32 => 6, Int64 => 7,
    Float16 => 8, Float32 => 9, Float64 => 10)

"""The element types GeoZL stores: 8 to 64 bit integers and 16 to 64 bit floats."""
const Sample = Union{UInt8,UInt16,UInt32,UInt64,Int8,Int16,Int32,Int64,
                     Float16,Float32,Float64}

const _NODATA_NONE = Cint(0)
const _NODATA_NAN = Cint(1)
const _NODATA_VALUE = Cint(2)

const _ERR_SIZE = 256

_errmsg(buf::Vector{UInt8}) =
    String(buf[1:something(findfirst(iszero, buf), length(buf) + 1)-1])

_bits_type(::Type{T}) where {T} =
    sizeof(T) == 1 ? UInt8 : sizeof(T) == 2 ? UInt16 : sizeof(T) == 4 ? UInt32 : UInt64

function _convert_nodata(::Type{T}, nodata) where {T}
    nodata isa Real || throw(ArgumentError("`nodata` must be a number, got $(repr(nodata))"))
    if nodata isa AbstractFloat && isnan(nodata) && !(T <: AbstractFloat)
        throw(ArgumentError("an $T raster cannot use NaN as nodata"))
    end
    value = try
        convert(T, nodata)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("nodata $nodata does not fit $T"))
    end
    isinf(value) && !isinf(nodata) && throw(ArgumentError("nodata $nodata does not fit $T"))
    return value
end

_fill_value(::Type{T}, nodata) where {T} =
    nodata === nothing ? T(NaN) : _convert_nodata(T, nodata)

# Dense samples of x. Missing values take the sentinel, or NaN in a float
# raster, so an array with missing values reads like a Rasters.jl one.
function _samples(x::AbstractArray, nodata)
    T = nonmissingtype(eltype(x))
    # Union{} is a subtype of everything, so an all-missing eltype passes the
    # subtype test and has to be named.
    (T !== Union{} && T <: Sample) ||
        throw(ArgumentError("GeoZL stores $(Sample), not $(eltype(x))"))
    eltype(x) === T && return x isa Array ? x : Array{T}(x)
    if nodata === nothing && !(T <: AbstractFloat)
        throw(ArgumentError("`x` has missing values; an $T raster needs `nodata` to mark them"))
    end
    fill = _fill_value(T, nodata)
    return T[ismissing(v) ? fill : v for v in x]
end

# The nodata mode and the sentinel's bits at T, as the C API takes them.
function _nodata(::Type{T}, a::Array{T}, nodata) where {T}
    if nodata === nothing
        mode = T <: AbstractFloat && any(isnan, a) ? _NODATA_NAN : _NODATA_NONE
        return mode, UInt64(0)
    end
    value = _convert_nodata(T, nodata)
    if value isa AbstractFloat && isnan(value)
        return _NODATA_NAN, UInt64(0)
    end
    T === Float16 &&
        throw(ArgumentError("a Float16 raster carries no nodata sentinel; NaN still works"))
    return _NODATA_VALUE, UInt64(reinterpret(_bits_type(T), value))
end

# Row width and plane count. The fastest-varying dimension is the row, so an
# array shaped (x, y) or (x, y, band) holds the same samples in the same order
# as Python's (y, x) or (band, y, x).
function _geometry(x::AbstractArray, width, planes)
    n = length(x)
    if width === nothing
        ndims(x) >= 2 || throw(ArgumentError("give `width` for a vector"))
        width = size(x, 1)
    end
    planes === nothing && (planes = ndims(x) >= 3 ? size(x, ndims(x)) : 1)
    width >= 1 || throw(ArgumentError("`width` must be at least 1, got $width"))
    planes >= 1 || throw(ArgumentError("`planes` must be at least 1, got $planes"))
    if planes > 1 && (n % planes != 0 || (n ÷ planes) % width != 0)
        throw(ArgumentError("$planes planes do not split $n samples into whole rows of $width"))
    end
    return UInt32(width), UInt32(planes)
end

"""
    Graph

A reusable compression graph from [`graph`](@ref). It holds native state, so it
is not thread-safe; build one per task.
"""
mutable struct Graph
    ptr::Ptr{Cvoid}
    method::String
    width::Int
    planes::Int
    eltype::DataType
    error::Union{Nothing,String}
    nodata::Any

    function Graph(ptr, method, width, planes, T, error, nodata)
        g = new(ptr, method, width, planes, T, error, nodata)
        return finalizer(g) do g
            g.ptr == C_NULL || ccall(_sym(:geozl_2d_graph_close_c), Cvoid, (Ptr{Cvoid},), g.ptr)
            g.ptr = C_NULL
        end
    end
end

function Base.show(io::IO, g::Graph)
    plane_word = g.planes == 1 ? "plane" : "planes"
    print(io, "GeoZL.Graph(", g.method, ", ", g.eltype, ", row width ", g.width, ", ",
          g.planes, " ", plane_word, ", ", something(g.error, "lossless"), ")")
end

"""
    graph(x, method; width=nothing, planes=nothing, error=nothing, nodata=nothing)

Build a reusable graph for `method`, such as `"planar>zigzag>pfor"`, from the
raster `x`. The graph fixes the element type and the geometry for every tile
compressed with it.

`width` defaults to `size(x, 1)` and `planes` to the last dimension of an array
with three or more. Row-based predictors reset at each plane; use
`width = nx * ny, planes = 1` to predict across bands.

`error` is `nothing` or `0` for lossless, a positive number for an absolute
bound (LINEAR), a percentage such as `"1%"` (LOG), or a full `LINEAR`, `LOG` or
`SQRT` recipe. For a lossy graph `x` fixes the domain later tiles are checked
against, so use data spanning the product.

`nodata` sets a sentinel; `missing` values in `x` are stored as it. NaN holes in
a float raster are found without asking.
"""
function graph(x::AbstractArray, method::AbstractString; width=nothing, planes=nothing,
               error=nothing, nodata=nothing)
    isempty(method) && throw(ArgumentError("`method` must be a recipe such as \"planar>zigzag>pfor\""))
    recipe = _normalize_error(error)
    a = _samples(x, nodata)
    T = eltype(a)
    mode, bits = _nodata(T, a, nodata)
    w, p = _geometry(x, width, planes)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    buf = zeros(UInt8, _ERR_SIZE)
    rc = GC.@preserve a ccall(_sym(:geozl_2d_graph_open_c), Cint,
        (Ref{Ptr{Cvoid}}, Cstring, UInt32, UInt32, Ptr{UInt8}, Cint, Cint, UInt64,
         Ptr{Cvoid}, Csize_t, Csize_t, Ptr{UInt8}, Csize_t),
        out, method, w, p, recipe === nothing ? C_NULL : recipe, _CODES[T], mode, bits,
        a, length(a), sizeof(T), buf, length(buf))
    rc == 0 || throw(GeoZLError("graph failed (method = $(repr(method))): " *
                                "$(_errmsg(buf)) (ZL error code $rc)"))
    return Graph(out[], String(method), Int(w), Int(p), T, recipe, nodata)
end

"""
    compress(x, g::Graph; coeffs=nothing) -> Vector{UInt8}

Compress the tile `x` with `g` and return the frame. Its element type must be
the graph's; its geometry is not checked. A lossy tile outside the domain the
graph was built from is refused.

`coeffs`, a vector of integer vectors, is attached to this frame. Read it back
with [`coeffs`](@ref); GeoZL assigns it no meaning.
"""
function compress(x::AbstractArray, g::Graph; coeffs=nothing)
    g.ptr == C_NULL && throw(GeoZLError("this graph is closed"))
    a = _samples(x, g.nodata)
    eltype(a) === g.eltype ||
        throw(ArgumentError("this graph was built for $(g.eltype), got $(eltype(a))"))
    bytes = sizeof(a)
    blob = coeffs === nothing ? UInt8[] : _pack_coeffs(coeffs)
    # Past the worst case; an incompressible tile still fits in 1.5x.
    cap = 1024 + bytes + bytes ÷ 2 + length(blob)
    dst = Vector{UInt8}(undef, cap)
    outsize = Ref{Csize_t}(0)
    buf = zeros(UInt8, _ERR_SIZE)
    rc = GC.@preserve g a dst blob begin
        if coeffs === nothing
            ccall(_sym(:geozl_2d_compress_graph_c), Cint,
                  (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{UInt8}, Csize_t, Ref{Csize_t},
                   Ptr{UInt8}, Csize_t),
                  g.ptr, a, length(a), dst, cap, outsize, buf, length(buf))
        else
            ccall(_sym(:geozl_2d_compress_coeffs_c), Cint,
                  (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t,
                   Ref{Csize_t}, Ptr{UInt8}, Csize_t),
                  g.ptr, a, length(a), blob, length(blob), dst, cap, outsize, buf, length(buf))
        end
    end
    rc == 0 || throw(GeoZLError("compress failed (method = $(repr(g.method))): " *
                                "$(_errmsg(buf)) (ZL error code $rc)"))
    return resize!(dst, outsize[])
end

function _frame_size(frame::Vector{UInt8}, max_output_size)
    if max_output_size !== nothing
        valid = max_output_size isa Real && !(max_output_size isa Bool) &&
                isfinite(max_output_size) && max_output_size >= 0
        valid || throw(ArgumentError("`max_output_size` must be a finite, non-negative number"))
    end
    dsize = GC.@preserve frame ccall(_sym(:geozl_2d_frame_dsize_c), Csize_t,
                                     (Ptr{UInt8}, Csize_t), frame, length(frame))
    dsize == 0 && throw(GeoZLError("decompress: unreadable frame"))
    if max_output_size !== nothing && dsize > max_output_size
        throw(GeoZLError("decompress: the frame declares $dsize bytes of output, above " *
                         "the $max_output_size allowed"))
    end
    return Int(dsize)
end

# Numeric output must be 8-byte aligned, which Julia's arrays are.
function _decompress!(out::Array, frame::Vector{UInt8}, verify::Bool)
    UInt(pointer(out)) % 8 == 0 || error("GeoZL: output buffer is not 8-byte aligned")
    outsize = Ref{Csize_t}(0)
    buf = zeros(UInt8, _ERR_SIZE)
    rc = GC.@preserve out frame ccall(_sym(:geozl_2d_decompress_c), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Cvoid}, Csize_t, Ref{Csize_t}, Cint, Ptr{UInt8}, Csize_t),
        frame, length(frame), out, sizeof(out), outsize, verify, buf, length(buf))
    rc == 0 || throw(GeoZLError("decompress failed: $(_errmsg(buf)) (ZL error code $rc)"))
    outsize[] == sizeof(out) ||
        throw(GeoZLError("decompress: the frame held $(outsize[]) bytes, not $(sizeof(out))"))
    return out
end

"""
    decompress(frame; verify=true, max_output_size=nothing) -> Vector{UInt8}
    decompress(T, frame, dims...; nodata=nothing, verify=true, max_output_size=nothing)

Decompress a frame. A frame does not record its element type or shape, so keep
them beside it. Without `T` the result is the raw native bytes; with it, an
`Array{T}` shaped `dims`, decoded in place. Samples equal to `nodata` become
`missing`.

Set `verify = false` to skip checksum verification, and `max_output_size` when
reading untrusted frames.
"""
function decompress(frame::AbstractVector{UInt8}; verify::Bool=true, max_output_size=nothing)
    f = frame isa Vector{UInt8} ? frame : Vector{UInt8}(frame)
    return _decompress!(Vector{UInt8}(undef, _frame_size(f, max_output_size)), f, verify)
end

function decompress(::Type{T}, frame::AbstractVector{UInt8}, dims::Integer...;
                    nodata=nothing, verify::Bool=true, max_output_size=nothing) where {T<:Sample}
    f = frame isa Vector{UInt8} ? frame : Vector{UInt8}(frame)
    dsize = _frame_size(f, max_output_size)
    dsize % sizeof(T) == 0 || throw(GeoZLError("decompress: $dsize bytes do not hold whole $T samples"))
    n = dsize ÷ sizeof(T)
    shape = isempty(dims) ? (n,) : Int.(dims)
    prod(shape) == n || throw(ArgumentError("`dims` hold $(prod(shape)) samples, the frame $n"))
    out = _decompress!(Array{T}(undef, shape), f, verify)
    nodata === nothing && return out
    sentinel = _convert_nodata(T, nodata)
    if sentinel isa AbstractFloat && isnan(sentinel)
        return Union{Missing,T}[isnan(v) ? missing : v for v in out]
    end
    return Union{Missing,T}[v === sentinel ? missing : v for v in out]
end

decompress(::Type{T}, frame::AbstractVector{UInt8}, dims::Tuple{Vararg{Integer}}; kw...) where {T<:Sample} =
    decompress(T, frame, dims...; kw...)

const _MAX_VECS = 255
const _MAX_COEFF_BYTES = 10_000

function _pack_coeffs(vectors)
    vectors isa AbstractVector ||
        throw(ArgumentError("`coeffs` must be a vector of integer vectors"))
    isempty(vectors) && throw(ArgumentError("`coeffs` must contain at least one vector"))
    length(vectors) > _MAX_VECS && throw(ArgumentError("`coeffs` supports at most $_MAX_VECS vectors"))
    used = 6
    rows = Vector{Vector{Int32}}(undef, length(vectors))
    for (j, v) in enumerate(vectors)
        v isa AbstractVector ||
            throw(ArgumentError("coeffs vector $j must be a vector of integers"))
        isempty(v) && throw(ArgumentError("coeffs vector $j is empty"))
        used += 4 + 4 * length(v)
        used > _MAX_COEFF_BYTES && throw(ArgumentError("coeffs exceeds the $_MAX_COEFF_BYTES-byte limit"))
        rows[j] = map(v) do c
            c isa Integer && !(c isa Bool) ||
                throw(ArgumentError("coefficient $(repr(c)) in vector $j is not an integer"))
            typemin(Int32) <= c <= typemax(Int32) ||
                throw(ArgumentError("coefficient $c in vector $j does not fit an Int32"))
            Int32(c)
        end
    end
    counts = UInt32[length(r) for r in rows]
    GC.@preserve rows counts begin
        pointers = Ptr{Int32}[pointer(r) for r in rows]
        need = ccall(_sym(:geozl_coeffs_size), Csize_t, (Ptr{UInt32}, Csize_t),
                     counts, length(counts))
        need == 0 && throw(ArgumentError("coeffs has an invalid shape"))
        blob = Vector{UInt8}(undef, need)
        buf = zeros(UInt8, _ERR_SIZE)
        written = ccall(_sym(:geozl_coeffs_pack), Csize_t,
                        (Ptr{UInt8}, Csize_t, Ptr{Ptr{Int32}}, Ptr{UInt32}, Csize_t,
                         Ptr{UInt8}, Csize_t),
                        blob, need, pointers, counts, length(counts), buf, length(buf))
    end
    written == 0 && throw(ArgumentError(something(_nonempty(_errmsg(buf)), "coeffs could not be packed")))
    return resize!(blob, written)
end

_nonempty(s::String) = isempty(s) ? nothing : s

"""
    coeffs(frame) -> Union{Nothing,Vector{Vector{Int32}}}

The coefficient vectors attached to `frame`, or `nothing` when it carries
none. Only the frame header is read.
"""
function coeffs(frame::AbstractVector{UInt8})
    f = frame isa Vector{UInt8} ? frame : Vector{UInt8}(frame)
    blobsize = Ref{Csize_t}(0)
    rc = GC.@preserve f ccall(_sym(:geozl_2d_frame_coeffs_c), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ref{Csize_t}), f, length(f), C_NULL, 0, blobsize)
    rc < 0 && return nothing
    rc == 0 || throw(GeoZLError("coeffs: unreadable frame (ZL error code $rc)"))
    blob = Vector{UInt8}(undef, blobsize[])
    rc = GC.@preserve f blob ccall(_sym(:geozl_2d_frame_coeffs_c), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ref{Csize_t}), f, length(f), blob, length(blob), blobsize)
    rc == 0 || throw(GeoZLError("coeffs: unreadable frame (ZL error code $rc)"))
    nvecs = Ref{Csize_t}(0)
    nvals = Ref{Csize_t}(0)
    parse_blob = (dst, ndst, counts, ncounts, outv, outn) -> GC.@preserve blob ccall(
        _sym(:geozl_coeffs_parse), Cint,
        (Ptr{UInt8}, Csize_t, Ptr{Int32}, Csize_t, Ptr{UInt32}, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}),
        blob, length(blob), dst, ndst, counts, ncounts, outv, outn)
    rc = parse_blob(C_NULL, 0, C_NULL, 0, nvecs, nvals)
    rc == 1 && return nothing
    rc == 0 || throw(GeoZLError("invalid GeoZL coefficient blob (code $rc)"))
    values = Vector{Int32}(undef, max(nvals[], 1))
    counts = Vector{UInt32}(undef, max(nvecs[], 1))
    rc = GC.@preserve values counts parse_blob(values, nvals[], counts, nvecs[], C_NULL, C_NULL)
    rc == 0 || throw(GeoZLError("invalid GeoZL coefficient blob (code $rc)"))
    out = Vector{Vector{Int32}}(undef, nvecs[])
    offset = 0
    for j in 1:nvecs[]
        out[j] = values[offset+1:offset+counts[j]]
        offset += counts[j]
    end
    return out
end
