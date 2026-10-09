# libgeozl, found once in __init__: GEOZL_LIB_PATH, then the library a checkout
# of the repository builds, then the artifact a release ships.

const libgeozl = Ref{Ptr{Cvoid}}(C_NULL)

_libname() = Sys.isapple() ? "libgeozl.dylib" :
             Sys.iswindows() ? "geozl.dll" : "libgeozl.so"

_artifact_metadata_url(version) =
    "https://github.com/asterisk-labs/geozl/releases/download/v$version/Artifacts.toml"

# A release publishes its Artifacts.toml beside the tarballs, since their hashes
# do not exist when the tag is made. It is fetched once into the depot, so later
# sessions load offline and without a request.
_metadata_cache(version) = joinpath(first(DEPOT_PATH), "geozl", "v$version", "Artifacts.toml")

# The Artifacts.toml to read and, when there is none, why.
function _artifacts_toml()
    local_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    isfile(local_toml) && return local_toml, ""
    version = pkgversion(@__MODULE__)
    version === nothing && return nothing, "the package has no version"
    cached = _metadata_cache(version)
    isfile(cached) && return cached, ""
    url = _artifact_metadata_url(version)
    try
        mkpath(dirname(cached))
        # Written aside first, so an interrupted download leaves no cache.
        mv(Downloads.download(url, cached * ".part"), cached; force = true)
    catch err
        return nothing, "could not fetch $url: $(sprint(showerror, err))"
    end
    return cached, ""
end

function _lib_path()
    env = get(ENV, "GEOZL_LIB_PATH", "")
    if !isempty(env)
        isfile(env) || error("GeoZL: GEOZL_LIB_PATH=$env does not exist")
        return env
    end
    checkout = joinpath(@__DIR__, "..", "..", "..", "core", "build", _libname())
    isfile(checkout) && return checkout
    toml, why = _artifacts_toml()
    if toml !== nothing
        # Releases list some platforms only.
        if Artifacts.artifact_meta("geozl", toml) === nothing
            why = "the release has no libgeozl for this platform"
        else
            base = LazyArtifacts.ensure_artifact_installed("geozl", toml)
            entries = readdir(base)
            if length(entries) == 1 && isdir(joinpath(base, entries[1]))
                base = joinpath(base, entries[1])
            end
            lib = joinpath(base, Sys.iswindows() ? "bin" : "lib", _libname())
            isfile(lib) && return lib
            why = "the artifact holds no $(_libname())"
        end
    end
    error("GeoZL: no libgeozl, $why. Build it with `make lib` in the GeoZL " *
          "repository or set GEOZL_LIB_PATH")
end

_sym(name::Symbol) = Libdl.dlsym(libgeozl[], name)

function __init__()
    libgeozl[] = Libdl.dlopen(_lib_path())
end
