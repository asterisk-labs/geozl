import struct

from openzl import ext as _ext

from .._ffi import _load_lib_full, _ptr, ffi

_CTID = 0x72D70E
_NAME = "geozl.lossless.blocked_transpose_zstd"

DEFAULT_BLOCK_SIZE = 2 * 1024 * 1024
MAX_BLOCK_SIZE = 64 * 1024 * 1024
_VERSION = 1
_WIDTHS = (1, 2, 4, 8)
_HEADER = struct.Struct("<BB2xIQ")

_desc = _ext.MultiInputCodecDescription(
    id=_CTID,
    name=_NAME,
    input_types=[_ext.Type.Numeric],
    singleton_output_types=[_ext.Type.Serial],
)


def _block_for_width(block_size, elt_width):
    block = int(block_size)
    if block < elt_width or block > MAX_BLOCK_SIZE:
        raise ValueError(
            f"{_NAME}: block_size must be between {elt_width} and "
            f"{MAX_BLOCK_SIZE} bytes, got {block}"
        )
    return block - block % elt_width


class _Encoder(_ext.CustomEncoder):
    def __init__(self, block_size):
        super().__init__()
        self._block_size = int(block_size)

    def multi_input_description(self):
        return _desc

    def encode(self, state):
        inp = state.inputs[0]
        n, elt = inp.num_elts, inp.elt_width
        if elt not in _WIDTHS:
            raise ValueError(f"{_NAME}: bad element width {elt}")
        block = _block_for_width(self._block_size, elt)
        full = _load_lib_full()
        cap = int(full.geozl_blocked_transpose_zstd_bound(n, elt, block))
        if n and cap == 0:
            raise ValueError(f"{_NAME}: output bound overflow")
        out = state.create_output(0, max(cap, 1), 1)
        written = ffi.new("size_t*")
        level = int(state.get_cparam(_ext.CParam.CompressionLevel))
        rc = full.geozl_blocked_transpose_zstd_encode(
            _ptr(out.mut_content.as_nparray()), cap, written,
            _ptr(inp.content.as_nparray()), n, elt, block, level)
        if rc:
            raise ValueError(f"{_NAME}: encode failed ({rc})")
        state.send_codec_header(_HEADER.pack(_VERSION, elt, block, n))
        out.commit(int(written[0]))


class BlockedTransposeZstdDecoder(_ext.CustomDecoder):
    def multi_input_description(self):
        return _desc

    def decode(self, state):
        src = state.singleton_inputs[0]
        raw = bytes(state.codec_header)
        if len(raw) != _HEADER.size:
            raise ValueError(f"{_NAME}: bad codec header, {len(raw)} bytes")
        version, elt, block, n = _HEADER.unpack(raw)
        if version != _VERSION or elt not in _WIDTHS:
            raise ValueError(f"{_NAME}: bad version or element width")
        _block_for_width(block, elt)
        if (n == 0) != (src.num_elts == 0):
            raise ValueError(f"{_NAME}: empty header/payload mismatch")
        out = state.create_output(0, n, elt)
        full = _load_lib_full()
        rc = full.geozl_blocked_transpose_zstd_decode(
            _ptr(out.mut_content.as_nparray()), n, elt, block,
            _ptr(src.content.as_nparray()), src.num_elts)
        if rc:
            raise ValueError(f"{_NAME}: corrupt stream ({rc})")
        out.commit(n)


class BlockedTransposeZstd:
    """Fuse byte shuffle and independent Zstd blocks into one serial stream."""

    def __init__(self, block_size=DEFAULT_BLOCK_SIZE):
        self._block_size = int(block_size)
        if self._block_size < 1 or self._block_size > MAX_BLOCK_SIZE:
            raise ValueError(
                f"block_size must be between 1 and {MAX_BLOCK_SIZE} bytes"
            )

    def __call__(self, compressor, successor):
        succ = (successor if isinstance(successor, _ext.GraphID)
                else successor.parameterize(compressor))
        node = compressor.register_custom_encoder(_Encoder(self._block_size))
        return compressor.build_static_graph(node, [succ], name=_NAME)


def bound(nb_elts, elt_width, block_size=DEFAULT_BLOCK_SIZE):
    """Worst-case payload size, excluding the OpenZL frame wrapper."""
    elt = int(elt_width)
    block = _block_for_width(block_size, elt)
    return int(_load_lib_full().geozl_blocked_transpose_zstd_bound(
        int(nb_elts), elt, block))


__all__ = [
    "BlockedTransposeZstd", "BlockedTransposeZstdDecoder",
    "DEFAULT_BLOCK_SIZE", "MAX_BLOCK_SIZE", "bound",
]
