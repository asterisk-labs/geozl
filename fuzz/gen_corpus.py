import pathlib
import sys

import numpy as np
import openzl.ext as zl

import geozl


_DISABLE = 2  # ZL_TernaryParam_disable


def _compress(node, arr, *, lossy=False):
    c = zl.Compressor()
    g = zl.graphs.Compress()(c)
    g = node(c, g)
    c.select_starting_graph(g)
    cc = zl.CCtx()
    cc.ref_compressor(c)
    cc.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    if lossy:
        cc.set_parameter(zl.CParam.ContentChecksum, _DISABLE)
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

    # One seed per decode path, a mutator will not invent a coherent header
    # for one curve out of another's.
    seeds = {
        "delta_w": _compress(geozl.lossless.DeltaW(w), tile16),
        "delta_n": _compress(geozl.lossless.DeltaN(w), tile16),
        "planar": _compress(geozl.lossless.Planar(w), tile16),
        "med": _compress(geozl.lossless.Med(w), tile16),
        "average": _compress(geozl.lossless.Average(w), tile16),
        "wp_static": _compress(geozl.lossless.WpStatic(w), tile16),
        "deinterleave": _compress(geozl.lossless.Deinterleave(), tile16),
        "delta_w_u8": _compress(geozl.lossless.DeltaW(w), tile8),
        "quant_linear_u16": _compress(
            geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=5", np.uint16), tile16,
            lossy=True),
        "quant_linear_f32": _compress(
            geozl.lossy.QuantLinear("LINEAR:MAX_ERROR=0.5", np.float32), tilef,
            lossy=True),
        "quant_sqrt_f32": _compress(
            geozl.lossy.QuantSqrt("SQRT:MAX_ERROR=0.5N,A=4,B=1", np.float32),
            tilef, lossy=True),
        "quant_log_f32": _compress(
            geozl.lossy.QuantLog("LOG:MAX_ERROR=5%", np.float32), tilef,
            lossy=True),
        "quant_log_wide_f32": _compress(
            geozl.lossy.QuantLog("LOG:MAX_ERROR=0.01%", np.float32), tilef,
            lossy=True),
        "quant_log_sub_f32": _compress(
            geozl.lossy.QuantLog("LOG:MAX_ERROR=1%", np.float32), tilesub,
            lossy=True),
    }

    for name, data in seeds.items():
        _decodes(data)
        (out / f"{name}.zl").write_bytes(data)
    print(f"wrote {len(seeds)} seed frames to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fuzz/corpus")