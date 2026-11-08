"""Write the golden corpus in bindings/python/test/golden.

The corpus holds one frame for every wire layout the current readers decode,
each written by the current code from a deterministic integer field. Run it
after a change to a wire layout, then commit the frames and the manifest.

    python tools/write_golden.py

Every frame is checked on the way out. It must decode back to its input
within the declared bound, and the header byte that names its layout must be
the one the entry claims, so a resolver change that silently picks another
layout fails here rather than shrinking the corpus.
"""

import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import openzl.ext as zl

import geozl
import geozl.lossless as gl

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "bindings" / "python" / "test" / "golden"
WRITER = f"geozl {geozl.__version__}"
NOTE = ("One frame per wire layout the current readers decode, written by "
        "tools/write_golden.py. See docs/compatibility.md before touching.")


def field(rows=64, cols=64, planes=1, noise=32, seed=1):
    """Smooth integer surface plus LCG noise, the same on every machine."""
    x = seed
    out = np.empty((planes, rows, cols), np.int64)
    for p in range(planes):
        for r in range(rows):
            for c in range(cols):
                x = (x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
                tri = abs((r * 5 + c * 3 + p * 17) % 128 - 64)
                out[p, r, c] = (1000 + 3 * r + 2 * c + ((r * c) >> 4) + 4 * tri
                                + (x >> 33) % noise)
    return out if planes > 1 else out[0]


def lanes_modes(frame, n, width, extra=0):
    """Per lane mode bytes of a lanes or planar_zigzag_lanes header."""
    at = frame.find(struct.pack("<QB", n, width))
    assert at >= 0, "no lanes header"
    lane0 = at + 9 + extra + 4
    return [frame[lane0 + 7 * k] for k in range(width)]


def sqrt_flags(frame, offset):
    """Flags byte of a quant_sqrt header, found by its offset field."""
    at = frame.find(struct.pack("<d", offset))
    assert at >= 10, "no quant_sqrt header"
    return frame[at - 9]


def recipe(arr, method, error=None, nodata=None, coeffs=None):
    g = geozl.graph(arr, method, error=error, nodata=nodata)
    return geozl.compress(arr, graph=g, coeffs=coeffs)


def low_level(build, arr):
    c = zl.Compressor()
    c.select_starting_graph(build(c))
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def entropy_tail(c):
    return zl.nodes.Zigzag()(c, zl.graphs.Entropy()(c))


FRAMES = []


def frame(name, codec, layout, graph, arr, data, error=None, nodata=None,
          coeffs=None, bound=0.0, check=None):
    FRAMES.append(dict(name=name, codec=codec, layout=layout, graph=graph,
                       arr=arr, data=data, error=error, nodata=nodata,
                       coeffs=coeffs, bound=bound, check=check))


def build():
    u16 = field().astype(np.uint16)
    u16_3p = field(planes=3).astype(np.uint16)
    i32 = (field() - 1500).astype(np.int32)
    u8 = (field() // 4 % 256).astype(np.uint8)
    u8_wide = (field(noise=256) % 64).astype(np.uint8)
    u8_few = (field() % 16).astype(np.uint8)
    u16_wide = field(noise=4096).astype(np.uint16)
    f32 = (field() / 8.0).astype(np.float32)
    w = u16.shape[1]

    # Spatial predictors through the recipe grid, both header forms.
    for pred, codec in (("planar", "planar_zigzag"), ("med", "med_zigzag"),
                        ("delta_n", "delta_n"), ("average", "average"),
                        ("wp_static", "wp_static")):
        slug = codec.replace("_", "-")
        g = f"{pred}>zigzag>entropy"
        frame(f"{slug}-uint16", codec, "one plane, short header", g, u16,
              recipe(u16, g))
        frame(f"{slug}-uint16-3p", codec, "three planes, long header", g,
              u16_3p, recipe(u16_3p, g))
    frame("delta-w-uint16", "delta_w", "width header",
          "delta_w>zigzag>entropy", u16, recipe(u16, "delta_w>zigzag>entropy"))

    # The predictors the grid always fuses, as standalone low-level nodes.
    for node, codec in ((gl.Planar, "planar"), (gl.Med, "med")):
        frame(f"{codec}-uint16", codec, "one plane, short header",
              f"{codec}>zigzag>entropy (low level)", u16,
              low_level(lambda c, n=node: n(w)(c, entropy_tail(c)), u16))
        frame(f"{codec}-uint16-3p", codec, "three planes, long header",
              f"{codec}>zigzag>entropy (low level)", u16_3p,
              low_level(lambda c, n=node: n(w, planes=3)(c, entropy_tail(c)),
                        u16_3p))
    pairs = u16.reshape(-1)
    frame("deinterleave-uint16", "deinterleave", "two lanes, no header",
          "deinterleave>delta_w>zigzag>compress (low level)", u16,
          low_level(lambda c: gl.Deinterleave()(c, gl.DeltaW(w // 2)(
              c, zl.nodes.Zigzag()(c, zl.graphs.Compress()(c)))), pairs))

    # Packers.
    frame("planar-zigzag-pfor-uint16", "planar_zigzag_pfor", "fused header",
          "planar>zigzag>pfor", u16, recipe(u16, "planar>zigzag>pfor"))
    frame("planar-zigzag-pfor-uint16-3p", "planar_zigzag_pfor",
          "fused header, three planes", "planar>zigzag>pfor", u16_3p,
          recipe(u16_3p, "planar>zigzag>pfor"))
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        a = (u8 if dt == np.uint8 else u16).astype(dt)
        name = np.dtype(dt).name
        frame(f"pfor-{name}", "pfor", f"{a.itemsize}-byte elements",
              "id>pfor", a, recipe(a, "id>pfor"))

    # Byte lanes, every width, both lane modes.
    frame("lanes-pfor-uint8", "lanes", "1 byte, PFOR lane", "id>lanes",
          u8_wide, recipe(u8_wide, "id>lanes"), check=("lanes", 0, [1]))
    frame("lanes-huffman-uint8", "lanes", "1 byte, Huffman lane", "id>lanes",
          u8_few, recipe(u8_few, "id>lanes"), check=("lanes", 0, [0]))
    for dt in (np.uint16, np.uint32, np.uint64):
        a = u16_wide.astype(dt)
        modes = [1, 0] + [0] * (a.itemsize - 2)
        frame(f"lanes-{np.dtype(dt).name}", "lanes",
              f"{a.itemsize} bytes, PFOR and Huffman lanes", "id>lanes", a,
              recipe(a, "id>lanes"), check=("lanes", 0, modes))
    frame("planar-zigzag-lanes-uint16", "planar_zigzag_lanes",
          "2 bytes, Huffman and PFOR lanes", "planar>zigzag>lanes", u16,
          recipe(u16, "planar>zigzag>lanes"), check=("lanes", 8, [0, 1]))

    # Plain OpenZL backends behind the id branch.
    frame("id-uint8", "none", "OpenZL entropy only", "id>entropy", u8,
          recipe(u8, "id>entropy"))
    frame("id-uint16-zstd", "none", "OpenZL zstd only", "id>zstd", u16,
          recipe(u16, "id>zstd"))

    # NoData, both forms.
    holes = i32.copy()
    holes[::7, ::5] = -9999
    frame("nodata-sentinel-int32", "nodata", "guarded form, sentinel",
          "planar>zigzag>transpose>entropy", holes,
          recipe(holes, "planar>zigzag>transpose>entropy", nodata=-9999),
          nodata=-9999)
    nan = f32.copy()
    nan[::7, ::5] = np.nan
    frame("nodata-nan-float32", "nodata", "plain form, NaN",
          "planar>zigzag>transpose>entropy", nan,
          recipe(nan, "planar>zigzag>transpose>entropy"))

    # Quantizers, every storage and arithmetic form.
    lin = "LINEAR:MAX_ERROR=2"
    frame("quant-linear-uint16", "quant_linear", "integer index",
          "planar>zigzag>pfor", u16, recipe(u16, "planar>zigzag>pfor", lin),
          error=lin, bound=2)
    frame("quant-linear-int16", "quant_linear", "signed integer index",
          "planar>zigzag>pfor", u16.astype(np.int16) - 1500,
          None, error=lin, bound=2)
    for dt in (np.float16, np.float32, np.float64):
        a = f32.astype(dt)
        frame(f"quant-linear-{np.dtype(dt).name}", "quant_linear",
              "float index", "planar>zigzag>transpose>entropy", a,
              recipe(a, "planar>zigzag>transpose>entropy", "LINEAR:MAX_ERROR=0.5"),
              error="LINEAR:MAX_ERROR=0.5", bound=0.5)
    log = "LOG:MAX_ERROR=1%"
    frame("quant-log-uint16", "quant_log", "integer values",
          "planar>zigzag>pfor", u16, recipe(u16, "planar>zigzag>pfor", log),
          error=log, bound=0.01)
    for dt in (np.float16, np.float32, np.float64):
        a = f32.astype(dt)
        frame(f"quant-log-{np.dtype(dt).name}", "quant_log", "float index",
              "planar>zigzag>transpose>entropy", a,
              recipe(a, "planar>zigzag>transpose>entropy", log),
              error=log, bound=0.01)
    sq = "SQRT:MAX_ERROR=2N,A=25,B=2"
    frame("quant-sqrt-uint16", "quant_sqrt", "integer index",
          "planar>zigzag>pfor", u16, recipe(u16, "planar>zigzag>pfor", sq),
          error=sq, check=("sqrt", 12.5, 0b001))
    tight = "SQRT:MAX_ERROR=0.12N,A=1,B=1"
    frame("quant-sqrt-values-uint8", "quant_sqrt", "integer values fallback",
          "planar>zigzag>entropy", u8_wide + 190,
          None, error=tight, check=("sqrt", 1.0, 0b011))
    for dt, rec, flags, layout in (
            (np.float16, "SQRT:MAX_ERROR=1N,A=100,B=1", 0b001, "float index"),
            (np.float32, "SQRT:MAX_ERROR=1N,A=100,B=1", 0b101,
             "float index, binary32 rebuild"),
            (np.float32, "SQRT:MAX_ERROR=1N,A=100,B=1", 0b001,
             "float index, binary64 rebuild"),
            (np.float64, "SQRT:MAX_ERROR=1N,A=100,B=1", 0b001, "float index")):
        wide = dt == np.float32 and flags == 0b001
        # Past 1e9 the binary32 charge outgrows a quarter of the step.
        a = (f32 * np.float32(1e10)).astype(dt) if wide else f32.astype(dt)
        suffix = "-b64" if wide else ""
        frame(f"quant-sqrt-{np.dtype(dt).name}{suffix}", "quant_sqrt", layout,
              "planar>zigzag>transpose>entropy", a,
              recipe(a, "planar>zigzag>transpose>entropy", rec),
              error=rec, check=("sqrt", 100.0, flags))

    # The frame header comment that carries coefficient vectors.
    co = [[7, -3, 11], [2, 5]]
    frame("coeffs-uint16", "coeffs", "frame header comment",
          "planar>zigzag>entropy", u16,
          recipe(u16, "planar>zigzag>entropy", coeffs=co), coeffs=co)


def verify(f):
    arr, raw = f["arr"], f["data"]
    if raw is None:
        raw = recipe(arr, f["graph"], f["error"], f["nodata"])
        f["data"] = raw
    out = geozl.decompress(raw).view(arr.dtype).reshape(arr.shape)
    if f["error"] is None:
        assert np.array_equal(out.view(np.uint8), arr.view(np.uint8)), f["name"]
    else:
        x, y = arr.astype(np.float64), out.astype(np.float64)
        if f["error"].startswith("LINEAR"):
            assert np.all(np.abs(x - y) <= f["bound"]), f["name"]
        elif f["error"].startswith("LOG"):
            assert np.all(np.abs(x - y) <= f["bound"] * np.abs(x)), f["name"]
        else:
            k, a, b = map(float, re.findall(r"[\d.]+", f["error"]))
            assert np.all(np.abs(x - y) <= k * np.sqrt(a + b * x)), f["name"]
        assert not np.array_equal(x, y), f"{f['name']} is lossless"
    check = f["check"]
    if check and check[0] == "lanes":
        got = lanes_modes(raw, arr.size, arr.itemsize, check[1])
        assert got == check[2], (f["name"], got)
    if check and check[0] == "sqrt":
        got = sqrt_flags(raw, check[1])
        assert got == check[2], (f["name"], bin(got))
    if f["coeffs"] is not None:
        assert geozl.coeffs(raw) == tuple(map(tuple, f["coeffs"])), f["name"]
    return out


def main():
    build()
    names = [f["name"] for f in FRAMES]
    assert len(names) == len(set(names)), "duplicate frame name"
    frames_dir = GOLDEN / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    entries = []
    for f in FRAMES:
        out = verify(f)
        (frames_dir / f"{f['name']}.zl").write_bytes(f["data"])
        entry = {
            "name": f["name"],
            "file": f"{f['name']}.zl",
            "codec": f["codec"],
            "layout": f["layout"],
            "graph": f["graph"],
            "error": f["error"],
            "nodata": f["nodata"],
            "dtype": f["arr"].dtype.name,
            "shape": list(f["arr"].shape),
            "written_by": WRITER,
            "sha256_decoded": hashlib.sha256(out.tobytes()).hexdigest(),
            "sha256_frame": hashlib.sha256(f["data"]).hexdigest(),
        }
        if f["coeffs"] is not None:
            entry["coeffs"] = f["coeffs"]
        entries.append(entry)
    manifest = {"note": NOTE, "frames": entries}
    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for e in entries:
        print(f"{e['name']:32} {e['codec']:20} {e['dtype']:8} {e['layout']}")


if __name__ == "__main__":
    main()
