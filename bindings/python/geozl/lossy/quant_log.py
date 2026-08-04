from .._codec import quantizer
from .._ffi import ffi, lib


def _plan(spec, dtype, src, nb_elts, params):
    err = ffi.new("char[]", 256)
    stats = ffi.new("quant_log_stats*")
    if lib.quant_log_scan(src, dtype, nb_elts, stats):
        return f"dtype {dtype} is not a type quant_log knows"
    if lib.quant_log_resolve(spec, dtype, stats, params, err, len(err)):
        return ffi.string(err).decode("utf-8", "replace")
    return None


QuantLog, QuantLogDecoder = quantizer(
    0x72D782, "geozl.lossy.quant_log", "quant_log", ("step",), _plan)

__all__ = ["QuantLog", "QuantLogDecoder"]
