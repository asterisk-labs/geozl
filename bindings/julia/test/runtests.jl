using GeoZL
using SHA
using Test

@test GeoZL._artifact_metadata_url(v"1.2.3") ==
      "https://github.com/asterisk-labs/geozl/releases/download/v1.2.3/Artifacts.toml"

# A smooth gradient of sines spanning [lo, hi], shaped (x, y).
function smooth(nx, ny, lo, hi)
    v = [sin(i / 9) + cos(j / 7) for i in 1:nx, j in 1:ny]
    lo .+ (v .- minimum(v)) ./ (maximum(v) - minimum(v)) .* (hi - lo)
end

base = smooth(40, 24, 0.0, 1.0)

# Each type with values that fit it exactly, extremes included.
cases = Dict(
    UInt8 => x -> round.(UInt8, x .* 255),
    UInt16 => x -> round.(UInt16, x .* 65535),
    UInt32 => x -> round.(UInt32, x .* 4294967295),
    UInt64 => x -> round.(UInt64, x .* 2.0^60),
    Int8 => x -> round.(Int8, x .* 255 .- 128),
    Int16 => x -> round.(Int16, x .* 65535 .- 32768),
    Int32 => x -> round.(Int32, x .* 4294967295 .- 2147483648),
    Int64 => x -> round.(Int64, x .* 2.0^62 .- 2.0^61),
    Float16 => x -> Float16.(round.(x .* 4096) .- 2048),
    Float32 => x -> Float32.(x .* 1000),
    Float64 => x -> x .* 1e300 .- 5e299,
)

roundtrip(x, method; kw...) =
    GeoZL.decompress(eltype(x), GeoZL.compress(x, GeoZL.graph(x, method; kw...)), size(x)...)

@testset "round trips" begin
    for (T, make) in cases
        x = make(base)
        @test roundtrip(x, "id>zstd") == x
    end
    for T in (UInt8, Int8, UInt16, Int16), method in ("planar>zigzag>pfor", "planar>zigzag>pivco",
                                                      "planar>zigzag>entropy", "med>zigzag>entropy")
        x = cases[T](base)
        @test roundtrip(x, method) == x
    end
    for T in (UInt32, Int32, UInt64, Int64)
        x = cases[T](base)
        @test roundtrip(x, "planar>zigzag>transpose>entropy") == x
    end
    # Float bits, not values: -0.0 and every NaN payload are samples too.
    x = cases[Float32](base)
    x[1] = -0.0f0
    back = roundtrip(x, "planar>zigzag>transpose>entropy")
    @test reinterpret(UInt32, back) == reinterpret(UInt32, x)
end

@testset "raw bytes" begin
    x = cases[Int16](base)
    frame = GeoZL.compress(x, GeoZL.graph(x, "planar>zigzag>pfor"))
    bytes = GeoZL.decompress(frame)
    @test bytes isa Vector{UInt8}
    @test bytes == reinterpret(UInt8, vec(x))
    @test GeoZL.decompress(Int16, frame, (40, 24)) == x
    @test GeoZL.decompress(Int16, frame) == vec(x)
end

@testset "layout" begin
    x = cases[UInt16](smooth(48, 32, 0.0, 1.0))
    g = GeoZL.graph(x, "planar>zigzag>pfor")
    @test (g.width, g.planes) == (48, 1)
    explicit = GeoZL.graph(vec(x), "planar>zigzag>pfor"; width = 48)
    @test GeoZL.compress(x, g) == GeoZL.compress(vec(x), explicit)

    cube = reshape(cases[UInt16](smooth(16, 36, 0.0, 1.0)), 16, 12, 3)
    gc = GeoZL.graph(cube, "planar>zigzag>pfor")
    @test (gc.width, gc.planes) == (16, 3)
    @test roundtrip(cube, "planar>zigzag>pfor") == cube

    # A view or a transpose is copied to dense samples first.
    @test roundtrip(view(x, 1:16, :), "planar>zigzag>pfor") == x[1:16, :]
    @test roundtrip(permutedims(x), "planar>zigzag>pfor") == permutedims(x)

    @test_throws ArgumentError GeoZL.graph(collect(UInt16, 1:64), "id>zstd")
    @test_throws ArgumentError GeoZL.graph(collect(UInt16, 1:64), "id>zstd"; width = 8, planes = 3)
    @test_throws ArgumentError GeoZL.decompress(UInt16, GeoZL.compress(x, g), 5, 5)
end

@testset "golden frames" begin
    dir = get(ENV, "GEOZL_GOLDEN_DIR",
              joinpath(@__DIR__, "..", "..", "python", "test", "golden"))
    manifest = joinpath(dir, "manifest.json")
    if isfile(manifest)
        # The manifest's entries are flat objects, so a pattern reads them
        # without a JSON dependency; the count check catches one it missed.
        text = read(manifest, String)
        entries = collect(eachmatch(r"\"file\":\s*\"([^\"]+)\"[^{}]*?\"sha256_decoded\":\s*\"([0-9a-f]+)\"[^{}]*?\"sha256_frame\":\s*\"([0-9a-f]+)\"", text))
        @test length(entries) == count(endswith(".zl"), readdir(joinpath(dir, "frames")))
        for m in entries
            frame = read(joinpath(dir, "frames", m[1]))
            @test bytes2hex(sha256(frame)) == m[3]
            @test bytes2hex(sha256(GeoZL.decompress(frame))) == m[2]
        end
    else
        @info "golden frames not found; set GEOZL_GOLDEN_DIR"
    end
end

@testset "errors" begin
    tile = cases[UInt16](smooth(32, 32, 0.0, 1.0))
    @test_throws GeoZLError GeoZL.graph(tile, "planar>nothing")
    @test_throws ArgumentError GeoZL.graph(tile, "")
    @test_throws ArgumentError GeoZL.graph(fill(true, 8, 8), "id>zstd")
    @test_throws ArgumentError GeoZL.graph(fill(1 + 2im, 8, 8), "id>zstd")
    @test_throws ArgumentError GeoZL.graph(fill(missing, 8, 8), "id>zstd"; nodata = 0)
    with_missing = Union{Missing,UInt8}[missing, 1]
    @test_throws ArgumentError GeoZL.graph(with_missing, "id>zstd"; nodata = -1, width = 2)
    g = GeoZL.graph(tile, "planar>zigzag>pfor")
    @test_throws ArgumentError GeoZL.compress(Int16.(tile .÷ 2), g)
    frame = GeoZL.compress(tile, g)
    @test_throws GeoZLError GeoZL.decompress(frame[1:end÷2])
    @test_throws GeoZLError GeoZL.decompress(collect(UInt8, 0:63))
    @test_throws GeoZLError GeoZL.decompress(frame; max_output_size = 100)
    for limit in (NaN, Inf, -1, "100", true)
        @test_throws ArgumentError GeoZL.decompress(frame; max_output_size = limit)
    end
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = NaN)
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = -1)
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = "none")
    flipped = copy(frame)
    flipped[end-4] ⊻= 0x01
    @test_throws GeoZLError GeoZL.decompress(flipped)
    @test_throws GeoZLError GeoZL.decompress(Float32, GeoZL.compress(tile[1:3, 1:1],
                                             GeoZL.graph(tile[1:3, 1:1], "id>zstd")))
end

@testset "bounded error" begin
    normalize = GeoZL._normalize_error
    @test normalize(nothing) === nothing
    @test normalize(0) === nothing
    @test normalize("0%") === nothing
    @test normalize("LINEAR:MAX_ERROR=0") === nothing
    @test normalize(2) == "LINEAR:MAX_ERROR=2"
    @test normalize(0.5) == "LINEAR:MAX_ERROR=0.5"
    @test normalize(1 // 2) == "LINEAR:MAX_ERROR=0.5"
    @test normalize("1%") == "LOG:MAX_ERROR=1%"
    @test normalize("SQRT:MAX_ERROR=2N") == "SQRT:MAX_ERROR=2N"
    @test_throws ArgumentError normalize(-1)
    @test_throws ArgumentError normalize(Inf)
    @test_throws ArgumentError normalize(true)
    @test_throws ArgumentError normalize("100%")
    @test_throws ArgumentError normalize("abc")

    x = Float32.(smooth(64, 48, 200.0, 3000.0) .+ 3 .* sin.(reshape(1:64*48, 64, 48)))
    back = roundtrip(x, "planar>zigzag>transpose>entropy"; error = 0.5)
    @test maximum(abs.(back .- x)) <= 0.5
    y = Float64.(x)
    back = roundtrip(y, "planar>zigzag>transpose>entropy"; error = "1%")
    @test all(abs.(back .- y) .<= 0.01 .* abs.(y))
    g = GeoZL.graph(x, "planar>zigzag>transpose>entropy"; error = 0.5)
    @test_throws GeoZLError GeoZL.compress(x .* 1f6, g)
end

@testset "nodata" begin
    x = Union{Missing,Int16}[round(Int16, v) for v in smooth(40, 30, -2000.0, 2000.0)]
    x[[5, 77, 1200]] .= missing
    g = GeoZL.graph(x, "planar>zigzag>pfor"; nodata = -9999)
    frame = GeoZL.compress(x, g)
    back = GeoZL.decompress(Int16, frame, size(x)...; nodata = -9999)
    @test isequal(back, x)
    @test GeoZL.decompress(Int16, frame, size(x)...)[[5, 77, 1200]] == fill(Int16(-9999), 3)
    @test_throws ArgumentError GeoZL.graph(x, "planar>zigzag>pfor")

    # Lossless keeps every NaN payload; a lossy frame writes them all back as
    # the first.
    f = smooth(40, 30, 0.0, 1.0)
    f[2] = NaN
    f[300] = reinterpret(Float64, 0x7FF00000000007A2)
    back = roundtrip(f, "planar>zigzag>transpose>entropy")
    @test reinterpret(UInt64, back) == reinterpret(UInt64, f)
    lossy = roundtrip(Float32.(f), "planar>zigzag>transpose>entropy"; error = 0.01)
    @test findall(isnan, lossy) == findall(isnan, f)

    m = Union{Missing,Float32}[v for v in Float32.(smooth(20, 20, 0.0, 100.0))]
    m[[3, 50]] .= missing
    back = GeoZL.decompress(Float32, GeoZL.compress(m, GeoZL.graph(m, "id>zstd")), 20, 20;
                            nodata = NaN)
    @test findall(ismissing, back) == findall(ismissing, m)

    @test_throws ArgumentError GeoZL.graph(fill(Float16(1), 8, 8), "id>zstd"; nodata = -1)
    @test_throws ArgumentError GeoZL.graph(fill(UInt16(1), 8, 8), "id>zstd"; nodata = NaN)
    @test_throws ArgumentError GeoZL.graph(fill(UInt16(1), 8, 8), "id>zstd"; nodata = 70000)
end

@testset "profile" begin
    x = cases[UInt16](smooth(64, 64, 0.0, 0.06))
    p = GeoZL.profile(x; reps = 1)
    @test p isa GeoZL.Profile
    @test length(p) > 5
    @test issorted([r.ratio for r in p]; rev = true)
    @test all(r -> startswith(r.graph, "planar>") || startswith(r.graph, "id>"), p)
    @test length(GeoZL.compress(x, GeoZL.graph(x, p[1].graph))) == p[1].bytes
    @test occursin("dec MB/s", sprint(show, MIME"text/plain"(), p))
    @test any(r -> startswith(r.graph, "med>"), GeoZL.profile(x; prior = nothing, reps = 1))
    @test_throws ArgumentError GeoZL.profile(x; prior = "bogus")
    @test_throws ArgumentError GeoZL.profile(x; prior = "")
    @test_throws ArgumentError GeoZL.profile(x; reps = 0)
end

@testset "coefficients" begin
    x = cases[UInt16](smooth(32, 32, 0.0, 1.0))
    g = GeoZL.graph(x, "planar>zigzag>pfor")
    vectors = [[1, 2, 3], [-5, 7], [typemax(Int32)]]
    frame = GeoZL.compress(x, g; coeffs = vectors)
    @test GeoZL.coeffs(frame) == vectors
    @test GeoZL.coeffs(frame) isa Vector{Vector{Int32}}
    @test GeoZL.decompress(UInt16, frame, size(x)...) == x
    @test GeoZL.coeffs(GeoZL.compress(x, g)) === nothing
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = Int[])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [Int[]])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [[1.5]])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [[2^31]])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = fill([1], 256))
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [collect(1:3000)])
end

# The frame Python writes for the same samples and settings, when GEOZL_PYTHON
# names a Python with geozl.
function python_frame(x, method; error = nothing, nodata = nothing)
    python = get(ENV, "GEOZL_PYTHON", "")
    isempty(python) && return nothing
    samples, frame = tempname(), tempname()
    a = x isa Array{<:GeoZL.Sample} ? x : GeoZL._samples(x, nodata)
    write(samples, a)
    arg(v) = v === nothing ? "None" : v isa AbstractString ? repr(v) :
             v isa AbstractFloat && isnan(v) ? "float('nan')" : string(v)
    dtype = lowercase(string(eltype(a)))
    script = """
    import numpy as np, geozl
    a = np.fromfile($(repr(samples)), '$dtype').reshape($(Tuple(reverse(size(a)))))
    g = geozl.graph(a, '$method', error=$(arg(error)), nodata=$(arg(nodata)))
    open($(repr(frame)), 'wb').write(geozl.compress(a, graph=g))
    """
    run(`$python -c $script`)
    out = read(frame)
    rm.((samples, frame); force = true)
    return out
end

@testset "frames match Python" begin
    if isempty(get(ENV, "GEOZL_PYTHON", ""))
        @info "GEOZL_PYTHON not set; skipping the comparison with Python"
    else
        holes = smooth(40, 30, 0.0, 1.0)
        holes[[2, 300, 400]] .= (NaN, reinterpret(Float64, 0x7FF00000000007A2), NaN)
        withmissing = Union{Missing,Int16}[round(Int16, v) for v in smooth(30, 30, -100.0, 100.0)]
        withmissing[[4, 9]] .= missing
        compare = [
            (cases[UInt16](smooth(64, 48, 0.0, 0.06)), "planar>zigzag>pfor", nothing, nothing),
            (reshape(cases[UInt16](smooth(32, 72, 0.0, 0.01)), 32, 24, 3), "planar>zigzag>pivco", nothing, nothing),
            (cases[UInt8](smooth(40, 40, 0.0, 1.0)), "med>zigzag>entropy", nothing, nothing),
            (Float32.(smooth(48, 40, 0.0, 500.0)), "planar>zigzag>transpose>entropy", 0.5, nothing),
            (smooth(48, 40, 1.0, 500.0), "planar>zigzag>transpose>entropy", "1%", nothing),
            (withmissing, "planar>zigzag>pfor", nothing, -9999),
            (Float16.(smooth(16, 16, -3.0, 3.0)), "id>zstd", nothing, nothing),
            (holes, "planar>zigzag>transpose>entropy", nothing, nothing),
        ]
        for (x, method, error, nodata) in compare
            g = GeoZL.graph(x, method; error, nodata)
            @test GeoZL.compress(x, g) == python_frame(x, method; error, nodata)
        end
    end
end
