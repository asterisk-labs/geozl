import pathlib
import sys

import numpy as np
import openzl.ext as zl

import geozl


def _compress(node, arr):
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = node(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    flat = np.ascontiguousarray(arr).reshape(-1)
    return bytes(cc.compress([zl.Input(zl.Type.Numeric, flat)]))


def _decodes(frame):
    """A seed the decoder rejects starts the fuzzer from an unreachable state,
    so it is worth a round trip here rather than a wasted run."""
    geozl.decompress(frame)


def main(outdir):
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # noise, not a gradient, so the encoded residuals are varied and give the
    # fuzzer more to mutate. Fixed seed keeps the corpus reproducible.
    rng = np.random.default_rng(0)
    w = 16
    tile16 = rng.integers(0, 1000, size=(w, w), dtype=np.uint16)
    tile8 = rng.integers(0, 256, size=(w, w), dtype=np.uint8)

    # twelve decades, positive, because the warped curves are about a tolerance
    # that follows the value and a narrow tile would not tell them apart
    tilef = (10.0 ** rng.uniform(-6, 6, size=(w, w))).astype(np.float32)

    # first row subnormal, where no grid meets a relative bound and the log
    # curve has to carry the values exactly
    tilesub = tilef.copy()
    tilesub[0] = np.arange(w, dtype=np.float64) * 1.40129846432481707e-45

    # The lossless seeds are built node by node. The lossy ones go through
    # geozl.compress instead, since the three quantizers have no Python encoder
    # classes and their parameters are resolved from a scan of the raster, which
    # only the high level entry does.
    #
    # One seed per decode path, since a mutator will not invent a coherent
    # header for one curve out of another's.
    # zstd serializes before the backend, so it takes whatever element width the
    # quantizer's index stream turns out to be. The entropy and transpose
    # terminals do not, and the width is not known here.
    lossy = "planar>zigzag>zstd"
    seeds = {
        "delta_w": _compress(geozl.lossless.DeltaW(w), tile16),
        "delta_n": _compress(geozl.lossless.DeltaN(w), tile16),
        "planar": _compress(geozl.lossless.Planar(w), tile16),
        "med": _compress(geozl.lossless.Med(w), tile16),
        "average": _compress(geozl.lossless.Average(w), tile16),
        "wp_static": _compress(geozl.lossless.WpStatic(w), tile16),
        "deinterleave": _compress(geozl.lossless.Deinterleave(), tile16),
        "delta_w_u8": _compress(geozl.lossless.DeltaW(w), tile8),
        "quant_linear_u16": geozl.compress(
            tile16, method=lossy, error="LINEAR:MAX_ERROR=5"),
        "quant_linear_f32": geozl.compress(
            tilef, method=lossy, error="LINEAR:MAX_ERROR=0.5"),
        "quant_sqrt_f32": geozl.compress(
            tilef, method=lossy, error="SQRT:MAX_ERROR=0.5N,A=4,B=1"),
        "quant_log_f32": geozl.compress(
            tilef, method=lossy, error="LOG:MAX_ERROR=5%"),
        "quant_log_wide_f32": geozl.compress(
            tilef, method=lossy, error="LOG:MAX_ERROR=0.01%"),
        "quant_log_sub_f32": geozl.compress(
            tilesub, method=lossy, error="LOG:MAX_ERROR=1%"),
    }

    for name, data in seeds.items():
        _decodes(data)
        (out / f"{name}.zl").write_bytes(data)
    print(f"wrote {len(seeds)} seed frames to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fuzz/corpus")