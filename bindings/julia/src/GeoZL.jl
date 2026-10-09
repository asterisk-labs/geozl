"""
    GeoZL

Julia bindings to GeoZL, the geospatial codecs for OpenZL. Build a graph once
with [`GeoZL.graph`](@ref), compress tiles with [`GeoZL.compress`](@ref), and
read them back with [`GeoZL.decompress`](@ref). [`GeoZL.profile`](@ref) ranks
recipes on a sample. The frames are the ones the Python and R packages write.
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
