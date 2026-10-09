"""
    GeoZL

Julia bindings to libgeozl, the geospatial codec extensions for OpenZL.
Build a graph once with [`GeoZL.graph`](@ref), compress tiles with
[`GeoZL.compress`](@ref), and read them back with [`GeoZL.decompress`](@ref).
[`GeoZL.profile`](@ref) ranks recipes on a sample.
"""
module GeoZL

using Artifacts
using Downloads
using LazyArtifacts
using Libdl
using Printf

export GeoZLError

include("lib.jl")
include("error.jl")
include("api.jl")
include("profile.jl")

end
