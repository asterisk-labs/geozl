import math
import pathlib
import struct
import sys

# Codes from qlog_dtype, only the ones the tiles below use.
U16, U32, I16, F32, F64 = 1, 2, 5, 9, 10

RECIPES = 9  # length of kRecipes in the harness


def rng(seed):
    """A plain LCG, so a corpus regenerated on another machine is the same one."""
    x = seed
    while True:
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        yield (x >> 33) / float(1 << 31)


def s2_dn(n):
    """Sentinel-2 surface reflectance as it ships, uint16 counts with a floor of
    zeros where the scene is masked."""
    r = rng(1)
    out = []
    for _ in range(n):
        v = next(r)
        out.append(0 if v < 0.08 else int(200 + 9000 * v * v))
    return struct.pack(f"<{n}H", *out)


def dem(n):
    """Elevation in metres, int16, crossing zero into bathymetry."""
    r = rng(2)
    out = []
    for i in range(n):
        base = 2000 * math.sin(i / 37.0) + 1500 * math.sin(i / 7.3)
        out.append(max(-400, min(8848, int(base + 200 * next(r)))))
    return struct.pack(f"<{n}h", *out)


def reflectance(n):
    """float32 in zero to one, with the exact zeros a masked scene carries."""
    r = rng(3)
    out = []
    for _ in range(n):
        v = next(r)
        out.append(0.0 if v < 0.05 else 0.02 + 0.9 * v * v)
    return struct.pack(f"<{n}f", *out)


def humidity(n):
    """Specific humidity, float32 over five decades. This is what the relative
    bound is for and where a flat one wastes most of its levels."""
    r = rng(4)
    out = [10.0 ** (-6.0 + 4.5 * next(r)) for _ in range(n)]
    return struct.pack(f"<{n}f", *out)


def kelvin(n):
    """float64 in a narrow band well away from zero, which is the band
    STORE=VALUES accepts."""
    r = rng(5)
    out = [250.0 + 80.0 * next(r) for _ in range(n)]
    return struct.pack(f"<{n}d", *out)


def anomaly(n):
    """float32 straddling zero, so the sign path and the floor flag both run."""
    r = rng(6)
    out = [(next(r) - 0.5) * 2.0 * 10.0 ** (-3.0 + 4.0 * next(r)) for _ in range(n)]
    return struct.pack(f"<{n}f", *out)


def counts(n):
    """uint32 photon counts, a wide integer type over three decades."""
    r = rng(7)
    out = [int(1 + 10.0 ** (5.0 * next(r))) for _ in range(n)]
    return struct.pack(f"<{n}I", *out)


TILES = [
    ("s2_dn", U16, s2_dn),
    ("dem", I16, dem),
    ("reflectance", F32, reflectance),
    ("humidity", F32, humidity),
    ("kelvin", F64, kelvin),
    ("anomaly", F32, anomaly),
    ("counts", U32, counts),
]

# Recipes worth reaching, plus the shapes that were holes in the sibling codec.
PARSE_SEEDS = [
    "LOG:MAX_ERROR=1%",
    "LOG:MAX_ERROR=0.5%",
    "LOG:MAX_ERROR=0.001%",
    "LOG:MAX_ERROR=1%,STORE=VALUES",
    "LOG:MAX_ERROR=1%,STORE=INDEX",
    "LOG:MAX_ERROR=1e-3%",
    "LOG:MAX_ERROR=100%",
    "LOG:MAX_ERROR=1",
    "LOG:MAX_ERROR=1%,MAX_ERROR=2%",
    "LOG:STORE=VALUES",
    "LOG:MAX_ERROR=1%,",
    "LOG:MAX_ERROR=0x1p0%",
    "LOG:MAX_ERROR=nan%",
    "LINEAR:MAX_ERROR=1",
]


def main(outdir):
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0

    # mode_parse. The first byte picks the mode, the rest is the recipe.
    for i, s in enumerate(PARSE_SEEDS):
        (out / f"parse_{i:02d}").write_bytes(b"\x00" + s.encode())
        n += 1

    # mode_roundtrip. Mode, recipe index, type, then the tile.
    for name, dtype, make in TILES:
        payload = make(512)
        for r in range(RECIPES):
            path = out / f"rt_{name}_{r}"
            path.write_bytes(bytes([2, r, dtype]) + payload)
            n += 1

    # mode_block. Mode, type, flags, step, then the stream. The step is a real
    # one so the block starts inside the range a frame carries and mutates out.
    for name, dtype, make in TILES:
        payload = make(256)
        for flags, step in ((3, 0.0144), (2, 0.0144), (1, 0.0072), (0, 0.0072)):
            path = out / f"blk_{name}_{flags}"
            path.write_bytes(
                bytes([1, dtype, flags]) + struct.pack("<d", step) + payload
            )
            n += 1

    print(f"{n} seeds in {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fuzz/corpus-quant-log")