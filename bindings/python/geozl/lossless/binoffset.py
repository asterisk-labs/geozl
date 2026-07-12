import struct

import numpy as np
from openzl import ext as _ext

from .._ffi import _ptr, ffi, lib

_CTID = 0x72D708
_NAME = "geozl.lossless.pco_binoffset"

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Numeric, _ext.Type.Numeric],
)

def _choose_bins_log(n, level):
    """pcodec choose_unoptimized_bins_log."""
    if n <= 0:
        return 0
    log_n = int(n).bit_length() - 1  # floor(log2(n))
    fast = max(0, log_n - 4)
    return level if level <= fast else fast + (level - fast) // 2


def _optimize_table(values, elt_width, level, meta_extra_bits):
    """pcodec binning in C: fine histogram (histograms.rs) then the boundary
    DP (bin_optimization.rs). Returns (lowers, offset_bits)."""
    v = np.ascontiguousarray(np.sort(values.astype(np.uint64)))
    n = v.size
    if n == 0:
        return np.zeros(0, np.uint64), np.zeros(0, np.uint8)
    n_bins_log = _choose_bins_log(n, level)
    cap = 1 << n_bins_log
    flo = np.zeros(cap, np.uint64)
    fup = np.zeros(cap, np.uint64)
    fcn = np.zeros(cap, np.uint32)
    nfine = lib.binoffset_histogram(
        ffi.cast("const uint64_t *", _ptr(v)), n, n_bins_log,
        ffi.cast("uint64_t *", _ptr(flo)),
        ffi.cast("uint64_t *", _ptr(fup)),
        ffi.cast("uint32_t *", _ptr(fcn)))
    meta = (elt_width + 1) * 8 + meta_extra_bits
    out_lo = np.zeros(255, np.uint64)
    out_ob = np.zeros(255, np.uint8)
    nb = lib.binoffset_optimize_table(
        ffi.cast("const uint64_t *", _ptr(flo)),
        ffi.cast("const uint64_t *", _ptr(fup)),
        ffi.cast("const uint32_t *", _ptr(fcn)), nfine, float(meta),
        ffi.cast("uint64_t *", _ptr(out_lo)),
        ffi.cast("uint8_t *", _ptr(out_ob)), 255)
    return out_lo[:nb].copy(), out_ob[:nb].copy()


def _pack_header(lowers, ob, elt_width):
    out = bytearray(struct.pack("<B", lowers.size))
    for l, o in zip(lowers.tolist(), ob.tolist()):
        out += int(l).to_bytes(8, "little")[:elt_width]
        out += struct.pack("<B", o)
    return bytes(out)


def _unpack_header(raw, elt_width):
    if len(raw) < 1:
        raise ValueError(f"{_NAME}: empty codec header")
    nb = raw[0]
    if nb == 0 or len(raw) != 1 + nb * (elt_width + 1):
        raise ValueError(f"{_NAME}: bad codec header size")
    lowers = np.zeros(256, np.uint64)
    ob = np.zeros(256, np.uint8)
    e = 1
    for b in range(nb):
        lowers[b] = int.from_bytes(raw[e:e + elt_width], "little")
        o = raw[e + elt_width]
        if o > 8 * elt_width:
            raise ValueError(f"{_NAME}: bad offset_bits in codec header")
        ob[b] = o
        e += elt_width + 1
    return lowers, ob, nb


class _Encoder(_ext.CustomEncoder):
    def __init__(self, level, meta_extra_bits):
        super().__init__()
        self._level = level
        self._meta_extra_bits = meta_extra_bits

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        src = inp.content.as_nparray()
        lowers, ob = _optimize_table(
            src.view(f"u{elt}") if n else np.zeros(1, f"u{elt}"), elt,
            self._level, self._meta_extra_bits)
        pad_lo = np.zeros(256, np.uint64)
        pad_ob = np.zeros(256, np.uint8)
        pad_lo[:lowers.size], pad_ob[:ob.size] = lowers, ob
        bins = state.create_output(0, n, 1)
        offs = state.create_output(1, n, elt)
        lib.binoffset_split_table(
            ffi.cast("uint8_t *", _ptr(bins.mut_content.as_nparray())),
            _ptr(offs.mut_content.as_nparray()), _ptr(src), n, elt,
            ffi.cast("const uint64_t *", _ptr(pad_lo)),
            ffi.cast("const uint8_t *", _ptr(pad_ob)), lowers.size)
        state.send_codec_header(_pack_header(lowers, ob, elt))
        bins.commit(n)
        offs.commit(n)


class BinOffsetDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        bins = state.singleton_inputs[0]
        offs = state.singleton_inputs[1]
        if bins.elt_width != 1 or bins.num_elts != offs.num_elts:
            raise ValueError(f"{_NAME}: corrupt stream shape")
        n, elt = offs.num_elts, offs.elt_width
        lowers, ob, nb = _unpack_header(bytes(state.codec_header), elt)
        out = state.create_output(0, n, elt)
        bad = lib.binoffset_join_table(
            _ptr(out.mut_content.as_nparray()),
            ffi.cast("const uint8_t *", _ptr(bins.content.as_nparray())),
            _ptr(offs.content.as_nparray()), n, elt,
            ffi.cast("const uint64_t *", _ptr(lowers)),
            ffi.cast("const uint8_t *", _ptr(ob)), nb)
        if bad:
            raise ValueError(f"{_NAME}: corrupt frame")
        out.commit(n)


class BinOffset:
    """Split a numeric stream into (bin, offset) pairs against a per-tile bin
    table from the boundary optimizer. Successors receive (bins, offsets); the
    table travels in the codec header. compression_level sets the fine-bin
    count; meta_extra_bits is the per-bin cost in the DP (models the FSE entry,
    the one deviation from pcodec's tANS cost)."""

    def __init__(self, compression_level=8, meta_extra_bits=24.0):
        self._level = int(compression_level)
        self._meta_extra_bits = float(meta_extra_bits)

    def __call__(self, compressor, bins_successor, offsets_successor):
        succ = []
        for s in (bins_successor, offsets_successor):
            if not isinstance(s, _ext.GraphID):
                s = s.parameterize(compressor)
            succ.append(s)
        node = compressor.register_custom_encoder(
            _Encoder(self._level, self._meta_extra_bits))
        return compressor.build_static_graph(node, succ, name=_NAME)