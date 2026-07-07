import pathlib
import sys

import numpy as np
import openzl.ext as zl

import geozl


def _compress(node, arr, drop_checksum=False):
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = node(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    if drop_checksum:
        cc.set_parameter(zl.CParam.ContentChecksum, 2)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def main(outdir):
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # noise, not a gradient, so the encoded residuals are varied and give the
    # fuzzer more to mutate. Fixed seed keeps the corpus reproducible.
    rng = np.random.default_rng(0)
    w = 16
    tile16 = rng.integers(0, 1000, size=(w, w), dtype=np.uint16)
    tile8 = rng.integers(0, 256, size=(w, w), dtype=np.uint8)

    seeds = {
        "delta_w": _compress(geozl.lossless.DeltaW(w), tile16),
        "delta_n": _compress(geozl.lossless.DeltaN(w), tile16),
        "planar": _compress(geozl.lossless.Planar(w), tile16),
        "med": _compress(geozl.lossless.Med(w), tile16),
        "average": _compress(geozl.lossless.Average(w), tile16),
        "wp_static": _compress(geozl.lossless.WpStatic(w), tile16),
        "deinterleave": _compress(geozl.lossless.Deinterleave(), tile16),
        "delta_w_u8": _compress(geozl.lossless.DeltaW(w), tile8),
        "quant_linear": _compress(
            geozl.lossy.QuantLinear(5, np.uint16), tile16, drop_checksum=True),
    }

    for name, data in seeds.items():
        (out / f"{name}.zl").write_bytes(data)
    print(f"wrote {len(seeds)} seed frames to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fuzz/corpus")