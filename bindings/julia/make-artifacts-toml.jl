# Prints the Artifacts.toml for the libgeozl tarballs a release attaches:
#
#     julia make-artifacts-toml.jl 0.19.0 lib-dist [platform] > Artifacts.toml
#
# The release workflow hashes the tarballs in lib-dist before publishing them,
# so the release and its metadata go out together; without a directory the
# tarballs are fetched from the published release. Without a version it reads
# the repository's VERSION.

using Downloads
using Pkg
using Pkg.Artifacts
using SHA

const REPO = "asterisk-labs/geozl"
const RELEASE = isempty(ARGS) ?
    strip(read(joinpath(@__DIR__, "..", "..", "VERSION"), String)) : strip(ARGS[1])
const LOCAL = length(ARGS) >= 2 ? ARGS[2] : nothing

const PLATFORMS = [
    (kv = ["arch = \"x86_64\"", "os = \"linux\"", "libc = \"glibc\""], name = "linux-x86_64"),
    (kv = ["arch = \"aarch64\"", "os = \"macos\""], name = "macos-arm64"),
]
const REQUESTED = length(ARGS) >= 3 ? Set(split(ARGS[3], ',')) :
                  Set(platform.name for platform in PLATFORMS)
const UNKNOWN = setdiff(REQUESTED, Set(platform.name for platform in PLATFORMS))
isempty(UNKNOWN) || error("unknown platform: $(join(sort!(collect(UNKNOWN)), ", "))")

function fetch(name)
    asset = "libgeozl-$RELEASE-$name.tar.gz"
    url = "https://github.com/$REPO/releases/download/v$RELEASE/$asset"
    archive = if LOCAL === nothing
        @info "Fetching" url
        Downloads.download(url)
    else
        path = joinpath(LOCAL, asset)
        isfile(path) || error("$path is missing")
        path
    end
    sha256_hex = bytes2hex(open(sha256, archive))
    # Pkg's own unpacker, so the tree hash is the one a download computes.
    tree = create_artifact() do dir
        Pkg.PlatformEngines.unpack(archive, dir)
    end
    return (url = url, sha256 = sha256_hex, tree = string(tree))
end

for platform in PLATFORMS
    platform.name in REQUESTED || continue
    info = fetch(platform.name)
    println("[[geozl]]")
    foreach(println, platform.kv)
    println("git-tree-sha1 = \"$(info.tree)\"")
    println("lazy = true")
    println()
    println("    [[geozl.download]]")
    println("    url = \"$(info.url)\"")
    println("    sha256 = \"$(info.sha256)\"")
    println()
end
