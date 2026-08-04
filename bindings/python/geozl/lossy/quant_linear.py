from .._codec import quantizer
from .._ffi import ffi, lib


def _plan(spec, dtype, src, nb_elts, params):
    err = ffi.new("char[]", 256)
    stats = ffi.new("quant_linear_stats*")
    if lib.quant_linear_scan(src, dtype, nb_elts, stats):
        return ("the stream holds no finite non-zero sample to cut a grid"
                " against")
    if lib.quant_linear_resolve(spec, dtype, stats, params, err, len(err)):
        return ffi.string(err).decode("utf-8", "replace")
    return None


QuantLinear, QuantLinearDecoder = quantizer(
    0x72D781, "geozl.lossy.quant_linear", "quant_linear", ("step",), _plan)

__all__ = ["QuantLinear", "QuantLinearDecoder"]
