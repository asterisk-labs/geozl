from openzl import ext as _ext

_CHECKSUM_DISABLE = 2   # OpenZL ternary: disable


def require_checksum_disabled(state, name):
    # Lossy codecs break the content hash, and OpenZL has it on by default.
    if state.get_cparam(_ext.CParam.ContentChecksum) != _CHECKSUM_DISABLE:
        raise ValueError(f"{name} is lossy, disable ContentChecksum on the CCtx")
