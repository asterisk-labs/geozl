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

# A primitive type GeoZL knows nothing of, and a struct with a padding byte.
primitive type Word32 32 end
struct Padded
    a::UInt8
    b::UInt16
end

roundtrip(x, method; kw...) =
    GeoZL.decompress(eltype(x), GeoZL.compress(x, GeoZL.graph(x, method; kw...)), size(x)...)

@testset "round trips" begin
    for (T, make) in cases
        x = make(base)
        @test roundtrip(x, "id>zstd") == x
    end
    for T in (UInt8, Int8, UInt16, Int16)
        x = cases[T](base)
        for method in ("planar>zigzag>pfor", "planar>zigzag>lanes", "planar>zigzag>entropy", "med>zigzag>entropy")
            @test roundtrip(x, method) == x
        end
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

@testset "frames own only their bytes" begin
    x = cases[UInt16](smooth(256, 256, 0.0, 0.01))
    frame = GeoZL.compress(x, GeoZL.graph(x, "planar>zigzag>pfor"))
    @test length(frame) < sizeof(x) ÷ 4
    # Not the compress bound the frame was written into.
    @test Base.summarysize(frame) == Base.summarysize(Vector{UInt8}(undef, length(frame)))
end

@testset "empty frames" begin
    x = Array{Int16}(undef, 8, 0)
    frame = GeoZL.compress(x, GeoZL.graph(x, "id>zstd"))
    @test isempty(GeoZL.decompress(frame; max_output_size = 0))
    @test isempty(GeoZL.decompress(frame; verify = false))
    @test GeoZL.decompress(Int16, frame, size(x)...) == x
    # An empty first dimension defaults the width to 0, as Python's last axis does.
    y = Array{Int16}(undef, 0, 8)
    g = GeoZL.graph(y, "id>zstd")
    @test g.width == 0
    @test GeoZL.decompress(Int16, GeoZL.compress(y, g), size(y)...) == y
    @test_throws ArgumentError GeoZL.graph(y, "id>zstd"; width = 0)
    @test_throws ArgumentError GeoZL.graph(Array{Int16}(undef, 0, 4, 3), "id>zstd")
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
        # No JSON parser is a test dependency, so a pattern reads the manifest's
        # flat entries. The count below fails if the pattern skips a frame.
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

@testset "graph fields" begin
    x = cases[UInt16](smooth(48, 32, 0.0, 1.0))
    g = GeoZL.graph(x, "planar>zigzag>pfor")
    @test g isa GeoZL.Graph{UInt16}
    @test eltype(g) === UInt16
    @test (g.method, g.width, g.planes, g.dtype, g.itemsize, g.error, g.nodata) ==
          ("planar>zigzag>pfor", 48, 1, "uint16", 2, nothing, nothing)
    @test repr(g) == "GeoZL.Graph(planar>zigzag>pfor, uint16, row width 48, 1 plane, lossless)"
    f = GeoZL.graph(Float32.(x), "planar>zigzag>transpose>entropy"; error = 2.0, planes = 2)
    @test (f.dtype, f.itemsize, f.planes, f.error) == ("float32", 4, 2, "LINEAR:MAX_ERROR=2.0")
    @test occursin("float32, row width 48, 2 planes, LINEAR:MAX_ERROR=2.0", repr(f))
end

@testset "element types" begin
    # Lossless graphs take any primitive element of 1, 2, 4 or 8 bytes but a
    # pointer, and complex numbers of them, as Python takes any native dtype.
    flags = [isodd(i + j) for i in 1:16, j in 1:8]
    @test roundtrip(flags, "id>zstd") == flags
    g = GeoZL.graph(flags, "id>zstd")
    @test (eltype(g), g.dtype, g.itemsize) == (Bool, "bool", 1)
    z = ComplexF32[complex(i, -j) for i in 1:16, j in 1:8]
    @test roundtrip(z, "planar>zigzag>transpose>entropy") == z
    @test GeoZL.graph(z, "id>zstd").dtype == "ComplexF32"
    c = Char['a' + (i + j) % 26 for i in 1:16, j in 1:8]
    @test roundtrip(c, "id>zstd") == c
    w = [reinterpret(Word32, UInt32(i * j)) for i in 1:16, j in 1:8]
    @test roundtrip(w, "planar>zigzag>transpose>entropy") == w
    @test GeoZL.decompress(Word32, GeoZL.compress(w, GeoZL.graph(w, "id>zstd")), 16, 8) == w
    h = ComplexF16[complex(i, j) for i in 1:16, j in 1:8]
    @test roundtrip(h, "id>zstd") == h
    @test length(GeoZL.profile(flags; reps = 1)) > 0
    @test_throws ArgumentError GeoZL.graph(fill(1 + 2im, 8, 8), "id>zstd")
    @test_throws ArgumentError GeoZL.graph(fill((1, 2, 3), 8, 8), "id>zstd")
    @test_throws ArgumentError GeoZL.graph(Any[1 2; 3 4], "id>zstd")
    @test_throws ArgumentError GeoZL.graph(fill(Ptr{Cvoid}(1), 8, 8), "id>zstd")
    @test_throws ArgumentError GeoZL.graph(fill(Padded(1, 2), 8, 8), "id>zstd")
    @test_throws ArgumentError GeoZL.decompress(Ptr{Cvoid}, UInt8[])
    @test_throws ArgumentError GeoZL.decompress(Padded, UInt8[])
    @test_throws ArgumentError GeoZL.graph(fill(Complex(true, false), 8, 8), "id>zstd")
    bytes = UInt8[0, 1, 2, 255]
    read_bool = GeoZL.decompress(Bool, GeoZL.compress(bytes, GeoZL.graph(bytes, "id>zstd"; width = 4)), 2, 2)
    @test read_bool == [false true; true true] && read_bool isa Matrix{Bool}
    @test_throws ArgumentError GeoZL.profile(fill(Padded(1, 2), 8, 8))
    # Lossy graphs and sentinels keep the eleven sample types.
    @test_throws ArgumentError GeoZL.graph(z, "id>zstd"; error = 1)
    @test_throws ArgumentError GeoZL.graph(flags, "id>zstd"; nodata = 0)
    @test_throws ArgumentError GeoZL.decompress(ComplexF32, GeoZL.compress(z, GeoZL.graph(z, "id>zstd"));
                                                nodata = 0)
    @test_throws ArgumentError GeoZL.decompress(Union{Missing,Int16}, UInt8[])
    @test_throws ArgumentError GeoZL.decompress(NTuple{3,UInt8}, UInt8[])
end

@testset "arguments" begin
    t = cases[UInt16](smooth(48, 32, 0.0, 1.0))
    m = "planar>zigzag>pfor"
    message(f) = try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end
    for bad in (3.7, 48.0, "48", true, 0, -1, 2^33, typemax(UInt32) + 1)
        @test_throws ArgumentError GeoZL.graph(vec(t), m; width = bad)
        @test_throws ArgumentError GeoZL.graph(t, m; planes = bad)
        @test_throws ArgumentError GeoZL.profile(vec(t); width = bad, reps = 1)
    end
    @test message(() -> GeoZL.graph(vec(t), m; width = 3.7)) ==
          "`width` must be an integer from 1 to 4294967295, got 3.7"
    @test GeoZL.graph(vec(t), m; width = UInt8(48)).width == 48
    @test GeoZL.graph(vec(t), m; width = big(48), planes = Int8(2)).planes == 2
    @test_throws ArgumentError GeoZL.graph(t, "id>zstd\0junk")
    @test_throws ArgumentError GeoZL.graph(Float32.(t), m; error = "LINEAR:MAX_ERROR=2\0junk")
    for bad in (true, "5", missing, [1], 1 + 2im, π, Inf, 1.5, -1)
        @test_throws ArgumentError GeoZL.graph(t, m; nodata = bad)
    end
    @test message(() -> GeoZL.graph(t, m; nodata = NaN)) == "a uint16 raster cannot use NaN as nodata"
    @test message(() -> GeoZL.graph(t, m; nodata = 70000)) == "nodata 70000 does not fit uint16"
    @test message(() -> GeoZL.graph(Float16.(t), m; nodata = -1)) ==
          "a half float carries no sentinel, NaN still works on one"
    for bad in (1, 0, "no", nothing)
        @test_throws ArgumentError GeoZL.decompress(GeoZL.compress(t, GeoZL.graph(t, m)); verify = bad)
        @test_throws ArgumentError GeoZL.profile(t; verify = bad, reps = 1)
    end
    for bad in (0, -1, 1.5, true, "5", 2^31)
        @test_throws ArgumentError GeoZL.profile(t; reps = bad)
    end
    @test_throws ArgumentError GeoZL.profile(t; prior = 5)
end

@testset "errors" begin
    tile = cases[UInt16](smooth(32, 32, 0.0, 1.0))
    # ZL's parameter_invalid and graph_invalid are invalid requests.
    @test_throws ArgumentError GeoZL.graph(tile, "planar>nothing")
    err = try GeoZL.graph(tile, "planar>nothing") catch e; e end
    @test err.msg == "GeoZL.graph failed (method = \"planar>nothing\"): unknown method " *
                     "\"planar>nothing\" (ZL error code 21)"
    @test_throws ArgumentError GeoZL.graph(tile, "")
    @test_throws ArgumentError GeoZL.graph(fill(missing, 8, 8), "id>zstd"; nodata = 0)
    with_missing = Union{Missing,UInt8}[missing, 1]
    @test_throws ArgumentError GeoZL.graph(with_missing, "id>zstd"; nodata = -1, width = 2)
    g = GeoZL.graph(tile, "planar>zigzag>pfor")
    @test_throws ArgumentError GeoZL.compress(Int16.(tile .÷ 2), g)
    frame = GeoZL.compress(tile, g)
    @test_throws GeoZLError GeoZL.decompress(frame[1:end÷2])
    junk = try GeoZL.decompress(collect(UInt8, 0:63)) catch e; e end
    @test junk isa GeoZLError && junk.code == 10
    @test junk.message == "GeoZL.decompress: unreadable frame (ZL error code 10)"
    @test sprint(showerror, junk) == "GeoZLError: " * junk.message
    @test GeoZLError(nothing, "x").code === nothing
    over = try GeoZL.decompress(frame; max_output_size = 100) catch e; e end
    @test over isa ArgumentError
    @test over.msg == "GeoZL.decompress: the frame declares 2048 bytes of output, above the 100 allowed"
    @test length(GeoZL.decompress(frame; max_output_size = 2048)) == 2048
    for limit in (NaN, Inf, -1, "100", true)
        @test_throws ArgumentError GeoZL.decompress(frame; max_output_size = limit)
    end
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = NaN)
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = -1)
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = "none")
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame; nodata = true)
    flipped = copy(frame)
    flipped[end-4] ⊻= 0x01
    corrupt = try GeoZL.decompress(flipped) catch e; e end
    @test corrupt isa GeoZLError && corrupt.code == 13
    @test startswith(corrupt.message, "GeoZL.decompress failed: ")
    # A corrupt frame can read back with graph_invalid, and is still a refused frame.
    invalid = nothing
    for i in eachindex(frame), bit in (0x01, 0x10, 0x80)
        bad = copy(frame)
        bad[i] ⊻= bit
        e = try GeoZL.decompress(bad); nothing catch e; e end
        if e !== nothing && endswith(sprint(showerror, e), "(ZL error code 30)")
            invalid = (bad, e)
            break
        end
    end
    @test invalid !== nothing
    @test invalid[2] isa GeoZLError && invalid[2].code == 30
    @test_throws GeoZLError GeoZL.decompress(UInt16, invalid[1], 32, 32)
    @test_throws ArgumentError GeoZL.decompress(Float32, GeoZL.compress(tile[1:3, 1:1],
                                                GeoZL.graph(tile[1:3, 1:1], "id>zstd")))
    @test_throws ArgumentError GeoZL.decompress(UInt16, frame, -32, -32)
    coeffs_junk = try GeoZL.coeffs(collect(UInt8, 0:63)) catch e; e end
    @test coeffs_junk isa GeoZLError && startswith(coeffs_junk.message, "GeoZL.coeffs: ")
end

varint(v::UInt64) = v < 0x80 ? UInt8[v] : vcat(UInt8(v & 0x7f | 0x80), varint(v >> 7))

# A frame of 300 UInt8 samples whose header declares `size` bytes in place of
# 300. The header checksum after the size is found by trying every byte until
# the size reads back.
function declaring(size::UInt64)
    x = ones(UInt8, 300)
    frame = GeoZL.compress(x, GeoZL.graph(x, "id>zstd"; width = 300))
    # Bytes 7 and 8 are the varint of 301, the size plus one, and byte 9 the checksum.
    @assert frame[7:8] == varint(UInt64(301))
    for check in 0x00:0xff
        f = vcat(frame[1:6], varint(size + 1), check, frame[10:end])
        e = try GeoZL.decompress(f; max_output_size = 0) catch e; e end
        e isa ArgumentError && occursin("declares", e.msg) && return f
    end
    error("no header checksum fits")
end

@testset "frames declaring more than Julia can allocate" begin
    # Past any address space, an Array Julia refuses, and past Int.
    for size in (UInt64(2)^62, UInt64(typemax(Int)), UInt64(2)^63 + 5)
        f = declaring(size)
        message = "GeoZL.decompress: the frame declares $size bytes of output, " *
                  "more than Julia can allocate"
        for read in (() -> GeoZL.decompress(f), () -> GeoZL.decompress(f; verify = false),
                     () -> GeoZL.decompress(UInt8, f))
            e = try read() catch e; e end
            @test e isa GeoZLError && e.code === nothing && e.message == message
        end
        @test_throws ArgumentError GeoZL.decompress(f; max_output_size = 1000)
    end
    e = try GeoZL.decompress(UInt16, declaring(UInt64(2)^62)) catch e; e end
    @test e isa GeoZLError && e.code === nothing
    # dims whose product wraps are the caller's mistake, not the frame's.
    x = UInt8[1, 2, 3, 4]
    f = GeoZL.compress(x, GeoZL.graph(x, "id>zstd"; width = 4))
    @test_throws ArgumentError GeoZL.decompress(UInt8, f, 2^62 + 1, 4)
    empty = GeoZL.compress(UInt8[], GeoZL.graph(UInt8[], "id>zstd"; width = 1))
    @test_throws ArgumentError GeoZL.decompress(UInt8, empty, 2^32, 2^32)
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
    # Floats are spelled as Python's repr and NumPy's str spell them.
    @test normalize(2.0) == "LINEAR:MAX_ERROR=2.0"
    @test normalize(1e20) == "LINEAR:MAX_ERROR=1e+20"
    @test normalize(1e-5) == "LINEAR:MAX_ERROR=1e-05"
    @test normalize(1e15) == "LINEAR:MAX_ERROR=1000000000000000.0"
    @test normalize(1e16) == "LINEAR:MAX_ERROR=1e+16"
    @test normalize(1e-4) == "LINEAR:MAX_ERROR=0.0001"
    @test normalize(1.2345678901234568e17) == "LINEAR:MAX_ERROR=1.2345678901234568e+17"
    @test normalize(5e-324) == "LINEAR:MAX_ERROR=5e-324"
    @test normalize(0.1f0) == "LINEAR:MAX_ERROR=0.1"
    @test normalize(1f6) == "LINEAR:MAX_ERROR=1e+06"
    @test normalize(1f-4) == "LINEAR:MAX_ERROR=1e-04"
    @test normalize(Float16(1000)) == "LINEAR:MAX_ERROR=1e+03"
    @test normalize(big"0.25") == "LINEAR:MAX_ERROR=0.25"
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
    # The library refuses a tile outside the domain as an invalid request.
    @test_throws ArgumentError GeoZL.compress(x .* 1f6, g)
    err = try GeoZL.compress(x .* 1f6, g) catch e; e end
    @test startswith(err.msg, "GeoZL.compress failed (method = \"planar>zigzag>transpose>entropy\"): ")
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

    # The eltype allows missing but no sample is, so no sentinel is needed.
    full = Union{Missing,Int16}[1 2; 3 4]
    back = GeoZL.decompress(Int16, GeoZL.compress(full, GeoZL.graph(full, "id>zstd")), 2, 2)
    @test back == full

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
    half = GeoZL.compress(fill(Float16(1), 8, 8), GeoZL.graph(fill(Float16(1), 8, 8), "id>zstd"))
    @test_throws ArgumentError GeoZL.decompress(Float16, half, 8, 8; nodata = -1)
    @test count(ismissing, GeoZL.decompress(Float16, half, 8, 8; nodata = NaN)) == 0
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
    @test p isa GeoZL.Profile{UInt16}
    @test (p.dims, p.dtype, p.raw_bytes, p.width, p.planes, p.prior, p.reps, p.error) ==
          ((64, 64), "uint16", 8192, 64, 1, "planar", 1, nothing)
    text = sprint(show, MIME"text/plain"(), p)
    @test occursin("dec MB/s", text)
    @test occursin("  dtype    uint16  [2 bytes/sample]\n", text)
    @test occursin("  profile  1 plane · row width 64 · planar + id · 1 rep · lossless\n", text)
    # When every recipe fails, the first reason is thrown.
    err = try GeoZL.profile(x; error = "NOPE:1", reps = 1) catch e; e end
    @test err isa ArgumentError
    @test startswith(err.msg, "GeoZL.profile failed: error \"NOPE:1\": ")
    @test any(r -> startswith(r.graph, "med>"), GeoZL.profile(x; prior = nothing, reps = 1))
    # "id" is the no-predictor path, reported as Python reports it.
    q = GeoZL.profile(x; prior = "id", reps = 1)
    @test q.prior == "none"
    @test occursin("no predictor", sprint(show, MIME"text/plain"(), q))
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
    @test GeoZL.compress(x, g; coeffs = ([1, 2, 3], [-5, 7], [typemax(Int32)])) == frame
    @test GeoZL.compress(x; graph = g, coeffs = vectors) == frame
    @test GeoZL.compress(x; graph = g) == GeoZL.compress(x, g)
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = ())
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = Int[])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [Int[]])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [[1.5]])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [[2^31]])
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = fill([1], 256))
    @test_throws ArgumentError GeoZL.compress(x, g; coeffs = [collect(1:3000)])
end

const PYTHON = get(ENV, "GEOZL_PYTHON", "")

# What a script prints in a Python with geozl, which GEOZL_PYTHON names.
# outcome(f) prints how f ends, as "ok" or the error's kind, code and message.
function python(script)
    prelude = """
    import numpy as np, geozl
    def outcome(f):
        try:
            f()
        except Exception as e:
            kind = 'argument' if isinstance(e, (ValueError, TypeError)) else 'library'
            print(kind, getattr(e, 'code', None))
            print(e, end='\\x1e')
        else:
            print('ok None', end='\\n\\x1e')
    """
    return read(`$PYTHON -c $(prelude * script)`, String)
end

# How each outcome(...) in a Python script ended, in Julia's words.
function python_outcomes(script)
    map(split(python(script), '\x1e')[1:end-1]) do record
        head, message = split(record, '\n'; limit = 2)
        kind, code = split(head)
        message = replace(message, r"^geozl\." => "GeoZL.",
                          r"\(method='([^']*)'\)" => s"(method = \"\1\")")
        (kind, kind == "library" && code != "None" ? parse(Int, code) : nothing, message)
    end
end

function outcome(f)
    try
        f()
        return ("ok", nothing, "")
    catch e
        e isa ArgumentError && return ("argument", nothing, e.msg)
        e isa GeoZLError && return ("library", e.code, e.message)
        rethrow()
    end
end

numpy_name(T) = get(Dict(Bool => "bool", ComplexF32 => "complex64"), T, lowercase(string(T)))

# A Python literal for an argument. Float32 and Float16 values keep their bits.
pyarg(v) = v === nothing ? "None" : v isa AbstractString ? repr(v) :
           v isa Float64 && isnan(v) ? "float('nan')" :
           v isa Float32 ? "np.uint32($(reinterpret(UInt32, v))).view(np.float32)" :
           v isa Float16 ? "np.uint16($(reinterpret(UInt16, v))).view(np.float16)" : repr(v)

pyfile(bytes) = (path = tempname(); write(path, bytes); repr(path))

# The samples of x as Python holds them, missing values marked as GeoZL does.
python_array(x, nodata = nothing) = (a = GeoZL._samples(x, nodata);
    "np.fromfile($(pyfile(a)), '$(numpy_name(eltype(a)))').reshape($(Tuple(reverse(size(a)))))")

# Python's frame for the same samples and settings, and its graph's fields.
function python_graph(x, method; width = nothing, planes = nothing, error = nothing, nodata = nothing)
    frame = tempname()
    printed = python("""
    a = $(python_array(x, nodata))
    g = geozl.graph(a, $(repr(method)), width=$(pyarg(width)), planes=$(pyarg(planes)),
                    error=$(pyarg(error)), nodata=$(pyarg(nodata)))
    open($(repr(frame)), 'wb').write(geozl.compress(a, graph=g))
    print(g.method, g.width, g.planes, g.dtype, g.itemsize, g.error, sep='\\n')
    """)
    return read(frame), split(chomp(printed), '\n')
end

fields(g) = [g.method, string(g.width), string(g.planes), g.dtype, string(g.itemsize),
             something(g.error, "None")]

if isempty(PYTHON)
    @info "GEOZL_PYTHON not set; skipping the comparison with Python"
else
    @testset "frames match Python" begin
        holes = smooth(40, 30, 0.0, 1.0)
        holes[[2, 300, 400]] .= (NaN, reinterpret(Float64, 0x7FF00000000007A2), NaN)
        withmissing = Union{Missing,Int16}[round(Int16, v) for v in smooth(30, 30, -100.0, 100.0)]
        withmissing[[4, 9]] .= missing
        compare = [
            (cases[UInt16](smooth(64, 48, 0.0, 0.06)), "planar>zigzag>pfor", nothing, nothing),
            (reshape(cases[UInt16](smooth(32, 72, 0.0, 0.01)), 32, 24, 3), "planar>zigzag>lanes", nothing, nothing),
            (cases[UInt8](smooth(40, 40, 0.0, 1.0)), "med>zigzag>entropy", nothing, nothing),
            (Float32.(smooth(48, 40, 0.0, 500.0)), "planar>zigzag>transpose>entropy", 0.5, nothing),
            (smooth(48, 40, 1.0, 500.0), "planar>zigzag>transpose>entropy", "1%", nothing),
            (withmissing, "planar>zigzag>pfor", nothing, -9999),
            (Float16.(smooth(16, 16, -3.0, 3.0)), "id>zstd", nothing, nothing),
            (holes, "planar>zigzag>transpose>entropy", nothing, nothing),
            (ComplexF32[complex(i, -j) for i in 1:16, j in 1:8], "planar>zigzag>transpose>entropy", nothing, nothing),
        ]
        for (x, method, error, nodata) in compare
            g = GeoZL.graph(x, method; error, nodata)
            @test GeoZL.compress(x, g) == first(python_graph(x, method; error, nodata))
        end
    end

    @testset "graphs match Python" begin
        flat = cases[UInt16](smooth(48, 32, 0.0, 0.06))
        cube = reshape(cases[UInt16](smooth(16, 36, 0.0, 1.0)), 16, 12, 3)
        float = Float32.(smooth(48, 40, 0.0, 500.0))
        compare = [
            (vec(flat), "planar>zigzag>pfor", (width = 48,)),
            (vec(flat), "planar>zigzag>pfor", (width = 48, planes = 2)),
            (flat, "planar>zigzag>pfor", (planes = 4,)),
            (cube, "planar>zigzag>lanes", (;)),
            (cube, "planar>zigzag>lanes", (width = 16 * 12, planes = 1)),
            (cube, "planar>zigzag>pfor", (width = 16, planes = 6)),
            (float, "planar>zigzag>transpose>entropy", (error = 0.5,)),
            (float, "planar>zigzag>transpose>entropy", (error = 2.0,)),
            (float, "planar>zigzag>transpose>entropy", (error = 2,)),
            (Float64.(float) .+ 1, "planar>zigzag>transpose>entropy", (error = "1%",)),
            ([isodd(i + j) for i in 1:16, j in 1:8], "id>zstd", (;)),
            (Array{UInt16}(undef, 0, 8), "id>zstd", (;)),
            (Array{Float32}(undef, 0, 8), "planar>zigzag>transpose>entropy", (;)),
        ]
        for (x, method, kw) in compare
            g = GeoZL.graph(x, method; kw...)
            frame, python_fields = python_graph(x, method; kw...)
            @test fields(g) == python_fields
            @test GeoZL.compress(x, g) == frame
        end
    end

    @testset "error text matches Python" begin
        errors = Any[2, 2.0, 0.5, 1e-5, 1e20, 0.1, 1e15, 1e16, 9999999999999998.0, 123456.789,
                     2.5e-7, 5e-324, floatmax(Float64), 0.1f0, 1f6, 1f-4, 999999.94f0,
                     Float16(0.1), Float16(1000), "1%", "+5%", "LINEAR:MAX_ERROR=0",
                     "SQRT:MAX_ERROR=2N", 0, 0.0]
        out = python("from geozl._2d import _normalize_error\n" *
                     join(["print(_normalize_error($(pyarg(e))))" for e in errors], "\n"))
        @test split(chomp(out), '\n') == [something(GeoZL._normalize_error(e), "None") for e in errors]
    end

    @testset "errors match Python" begin
        tile = cases[UInt16](smooth(32, 32, 0.0, 1.0))
        float = Float32.(smooth(64, 48, 200.0, 3000.0))
        lossy = GeoZL.graph(float, "planar>zigzag>transpose>entropy"; error = 0.5)
        frame = GeoZL.compress(tile, GeoZL.graph(tile, "planar>zigzag>pfor"))
        flipped = copy(frame)
        flipped[end-4] ⊻= 0x01
        julia = [
            outcome(() -> GeoZL.graph(tile, "planar>nothing")),
            outcome(() -> GeoZL.compress(float .* 1f6, lossy)),
            outcome(() -> GeoZL.decompress(frame; max_output_size = 100)),
            outcome(() -> GeoZL.decompress(flipped)),
            outcome(() -> GeoZL.decompress(frame[1:end÷2])),
            outcome(() -> GeoZL.decompress(collect(UInt8, 0:63))),
            outcome(() -> GeoZL.profile(tile; error = "NOPE:1", reps = 1)),
        ]
        py = python_outcomes("""
        t = $(python_array(tile))
        x = $(python_array(float))
        g = geozl.graph(x, 'planar>zigzag>transpose>entropy', error=0.5)
        f = open($(pyfile(frame)), 'rb').read()
        flipped = open($(pyfile(flipped)), 'rb').read()
        junk = bytes(range(64))
        outcome(lambda: geozl.graph(t, 'planar>nothing'))
        outcome(lambda: geozl.compress(x * np.float32(1e6), graph=g))
        outcome(lambda: geozl.decompress(f, max_output_size=100))
        outcome(lambda: geozl.decompress(flipped))
        outcome(lambda: geozl.decompress(f[:len(f) // 2]))
        outcome(lambda: geozl.decompress(junk))
        outcome(lambda: geozl.profile(t, error='NOPE:1', reps=1))
        """)
        @test length(py) == length(julia)
        for (j, p) in zip(julia, py)
            @test j == p
        end
    end

    @testset "profiles match Python" begin
        tile = cases[UInt16](smooth(48, 32, 0.0, 0.06))
        float = Float32.(smooth(48, 40, 0.0, 500.0))
        for (x, kw) in ((tile, (prior = "planar",)), (tile, (prior = nothing,)),
                        (float, (prior = "planar", error = 0.5)))
            p = GeoZL.profile(x; reps = 1, kw...)
            out = python("""
            p = geozl.profile($(python_array(x)), prior=$(pyarg(kw.prior)),
                              error=$(pyarg(get(kw, :error, nothing))), reps=1)
            print(p.dtype, p.raw_bytes, p.width, p.planes, p.prior, p.reps, p.error)
            for r in p:
                print(r['graph'], r['bytes'])
            """)
            settings, rows... = split(chomp(out), '\n')
            @test split(settings) == string.([p.dtype, p.raw_bytes, p.width, p.planes,
                                               something(p.prior, "None"), p.reps,
                                               something(p.error, "None")])
            @test [(r.graph, string(r.bytes)) for r in p] == [Tuple(split(r)) for r in rows]
        end
    end

    @testset "typed reads match Python" begin
        for x in (Int64[typemin(Int64), typemax(Int64), 0, -1, 2^53 + 1, -(2^53 + 1), 2^62],
                  UInt64[0, typemax(UInt64), UInt64(2)^63, UInt64(2)^53 + 1, typemax(UInt64) - 1])
            for method in ("id>zstd", "planar>zigzag>transpose>entropy")
                frame = GeoZL.compress(x, GeoZL.graph(x, method; width = length(x)))
                out = python("""
                f = open($(pyfile(frame)), 'rb').read()
                print(*geozl.decompress(f).view('$(numpy_name(eltype(x)))').tolist())
                """)
                @test GeoZL.decompress(eltype(x), frame) == parse.(eltype(x), split(out)) == x
            end
        end
        bytes = collect(UInt8, 0:255)
        frame = GeoZL.compress(bytes, GeoZL.graph(bytes, "id>zstd"; width = 256))
        out = python("""
        f = open($(pyfile(frame)), 'rb').read()
        print(*geozl.decompress(f).view(np.bool_).tolist())
        """)
        @test GeoZL.decompress(Bool, frame) == (split(out) .== "True")
    end
end
