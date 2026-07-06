from openzl import ext as _ext

_CHECKSUM_DISABLE = 2   # OpenZL ternary parameter, disable


def require_checksum_disabled(state, name):
    # A lossy codec breaks the content hash, which OpenZL has on by default.
    if state.get_cparam(_ext.CParam.ContentChecksum) != _CHECKSUM_DISABLE:
        raise ValueError(f"{name} is lossy, disable ContentChecksum on the CCtx")
