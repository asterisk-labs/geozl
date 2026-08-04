from .._codec import quantizer
from .._ffi import ffi, lib


def _plan(spec, dtype, src, nb_elts, params):
    err = ffi.new("char[]", 256)
    max_abs = ffi.new("double*")
    any_negative = ffi.new("int*")
    if lib.quant_linear_scan(src, dtype, nb_elts, max_abs, any_negative):
        return ("the stream holds no finite non-zero sample to cut a grid"
                " against")
    if lib.quant_linear_resolve(spec, dtype, max_abs[0], any_negative[0],
                                params, err, len(err)):
        return ffi.string(err).decode("utf-8", "replace")
    return None


QuantLinear, QuantLinearDecoder = quantizer(
    0x72D781, "geozl.lossy.quant_linear", "quant_linear", ("step",), _plan)

__all__ = ["QuantLinear", "QuantLinearDecoder"]
