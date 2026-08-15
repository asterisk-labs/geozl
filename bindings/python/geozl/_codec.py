import struct

from openzl import ext as _ext

from ._dtype import dtype_code, dtype_width
from ._ffi import _ptr, ffi, lib

# Little-endian width, followed by the plane count when above one.
_HEADER = struct.Struct("<I")
_HEADER_PLANES = struct.Struct("<II")

_CHECKSUM_DISABLE = 2  # ZL_TernaryParam_disable


def row_width_declared(width, nb_elts):
    """Return the row width stored in a predictor header."""
    return nb_elts if width == 0 or width > nb_elts else width


def planes_declared(planes, width, nb_elts):
    """Return the plane count when it splits the stream into whole rows, else 1."""
    if planes <= 1 or width == 0 or nb_elts == 0:
        return 1
    if nb_elts % planes:
        return 1
    per = nb_elts // planes
    if per == 0 or per % width:
        return 1
    return planes


def _per_plane(fn, dst, src, width, nb_elts, elt, planes):
    """Run a kernel once per plane, returning the first non-zero result."""
    per = nb_elts // planes
    for pl in range(planes):
        at = pl * per * elt
        rc = fn(dst + at, src + at, width, per, elt)
        if rc:
            return rc
    return 0


def _unpack_geometry(name, raw, nb_elts):
    """Read width and planes off a predictor header, as the C reader does."""
    if len(raw) == _HEADER.size:
        (width,), planes = _HEADER.unpack(raw), 1
    elif len(raw) == _HEADER_PLANES.size:
        width, planes = _HEADER_PLANES.unpack(raw)
    else:
        raise ValueError(f"{name}: bad codec header, {len(raw)} bytes where the "
                         f"frame carries {_HEADER.size} or "
                         f"{_HEADER_PLANES.size}")
    if planes == 0 or width == 0:
        raise ValueError(f"{name}: codec header declares width {width} over "
                         f"{planes} planes")
    if planes > 1 and (nb_elts % planes or (nb_elts // planes) % width):
        raise ValueError(f"{name}: {planes} planes do not split {nb_elts} "
                         f"samples into whole rows of {width}")
    return width, planes


def spatial_predictor(ctid, name, encode, decode, supports_planes=True):
    """Create OpenZL bindings for a spatial predictor."""
    enc = getattr(lib, encode)
    dec = getattr(lib, decode)
    short = name.rsplit(".", 1)[-1]

    desc = _ext.MultiInputCodecDescription(
        id=ctid,
        name=name,
        input_types=[_ext.Type.Numeric],
        singleton_output_types=[_ext.Type.Numeric],
    )

    class Encoder(_ext.CustomEncoder):
        def __init__(self, width, planes=1):
            super().__init__()
            self._width = int(width)
            self._planes = int(planes) if supports_planes else 1

        def multi_input_description(self):
            return desc

        def encode(self, state):
            inp = state.inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            width = row_width_declared(self._width, n)
            nb_planes = planes_declared(self._planes, width, n)
            out = state.create_output(0, n, elt)
            # n == 0 is a valid empty stream, the kernel rejects it as a geometry
            if n and _per_plane(enc, _ptr(out.mut_content.as_nparray()),
                                _ptr(inp.content.as_nparray()), width, n, elt,
                                nb_planes):
                raise ValueError(
                    f"{name}: width {self._width} does not tile {n} samples")
            state.send_codec_header(
                _HEADER_PLANES.pack(width, nb_planes) if nb_planes > 1
                else _HEADER.pack(width))
            out.commit(n)

    class Decoder(_ext.CustomDecoder):
        def multi_input_description(self):
            return desc

        def decode(self, state):
            inp = state.singleton_inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            width, nb_planes = _unpack_geometry(name, bytes(state.codec_header),
                                                n)
            out = state.create_output(0, n, elt)
            if n and _per_plane(dec, _ptr(out.mut_content.as_nparray()),
                                _ptr(inp.content.as_nparray()), width, n, elt,
                                nb_planes):
                raise ValueError(f"{name}: bad row width in codec header")
            out.commit(n)

    class Node:
        def __init__(self, width, planes=1):
            self._width = int(width)
            self._planes = int(planes)
            if not supports_planes and self._planes != 1:
                raise ValueError(f"{name} does not support planes")

        def __call__(self, compressor, successor):
            if not isinstance(successor, _ext.GraphID):
                successor = successor.parameterize(compressor)
            node = compressor.register_custom_encoder(
                Encoder(self._width, self._planes))
            return compressor.build_static_graph(node, [successor], name=name)

    Encoder.__name__ = Encoder.__qualname__ = f"_{short}_encoder"
    Decoder.__name__ = Decoder.__qualname__ = f"_{short}_decoder"
    Node.__name__ = Node.__qualname__ = f"_{short}_node"
    return Node, Decoder


def quantizer(ctid, name, prefix, doubles, no_grid, extra=()):
    """Create OpenZL bindings for a lossy quantizer."""
    parse = getattr(lib, f"{prefix}_parse")
    scan = getattr(lib, f"{prefix}_scan")
    resolve = getattr(lib, f"{prefix}_resolve")
    enc = getattr(lib, f"{prefix}_encode")
    dec = getattr(lib, f"{prefix}_decode")
    params = f"{prefix}_params*"
    stats = f"{prefix}_stats*"
    short = name.rsplit(".", 1)[-1]

    header = struct.Struct("<BB" + "d" * len(doubles))

    desc = _ext.MultiInputCodecDescription(
        id=ctid,
        name=name,
        input_types=[_ext.Type.Numeric],
        singleton_output_types=[_ext.Type.Numeric],
    )

    class Encoder(_ext.CustomEncoder):
        def __init__(self, spec, dtype):
            super().__init__()
            self._spec = spec
            self._dtype = dtype

        def multi_input_description(self):
            return desc

        def encode(self, state):
            # Lossy, so the content hash OpenZL keeps by default would fail.
            if state.get_cparam(
                    _ext.CParam.ContentChecksum) != _CHECKSUM_DISABLE:
                raise ValueError(
                    f"{name} is lossy, disable ContentChecksum on the CCtx")
            inp = state.inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            if dtype_width(self._dtype) != elt:
                raise ValueError(f"{name}: dtype {self._dtype} does not match "
                                 f"{elt}-byte samples")
            src = _ptr(inp.content.as_nparray())
            sc = ffi.new(stats)
            if scan(src, self._dtype, n, sc):
                raise ValueError(
                    f"{name}: {no_grid.format(dtype=self._dtype)}")
            p = ffi.new(params)
            err = ffi.new("char[]", 256)
            if resolve(self._spec, self._dtype, sc, *extra, p, err, len(err)):
                raise ValueError(
                    f"{name}: {ffi.string(err).decode('utf-8', 'replace')}")
            out = state.create_output(0, n, elt)
            if enc(_ptr(out.mut_content.as_nparray()), src, p, self._dtype, n):
                raise ValueError(
                    f"{name}: resolved parameters its own kernels reject")
            state.send_codec_header(header.pack(
                self._dtype, p.flags, *(getattr(p, f) for f in doubles)))
            out.commit(n)

    class Decoder(_ext.CustomDecoder):
        def multi_input_description(self):
            return desc

        def decode(self, state):
            inp = state.singleton_inputs[0]
            n, elt = inp.num_elts, inp.elt_width
            raw = bytes(state.codec_header)
            if len(raw) != header.size:
                raise ValueError(f"{name}: bad codec header, {len(raw)} bytes "
                                 f"where the frame carries {header.size}")
            dtype, flags, *vals = header.unpack(raw)
            # The kernel writes at the width this byte names.
            if dtype_width(dtype) != elt:
                raise ValueError(f"{name}: bad dtype in codec header")
            p = ffi.new(params)
            p.flags = flags
            for field, value in zip(doubles, vals, strict=True):
                setattr(p, field, value)
            out = state.create_output(0, n, elt)
            # The rest is params_ok, inside the kernel, read by both ends.
            if dec(_ptr(out.mut_content.as_nparray()),
                   _ptr(inp.content.as_nparray()), p, dtype, n):
                raise ValueError(
                    f"{name}: the kernel refused this codec header")
            out.commit(n)

    class Node:
        def __init__(self, recipe, dtype):
            if not isinstance(recipe, str) or not recipe:
                raise ValueError(
                    f"{name} takes a recipe string, got {recipe!r}")
            code = dtype_code(dtype)
            if code is None:
                raise ValueError(f"{name} has no kernel for dtype {dtype}")
            spec = ffi.new(f"{prefix}_spec*")
            err = ffi.new("char[]", 256)
            if parse(recipe.encode("utf-8"), spec, err, len(err)):
                raise ValueError(ffi.string(err).decode("utf-8", "replace"))
            self._spec = spec
            self._dtype = code

        def __call__(self, compressor, successor):
            if not isinstance(successor, _ext.GraphID):
                successor = successor.parameterize(compressor)
            node = compressor.register_custom_encoder(
                Encoder(self._spec, self._dtype))
            return compressor.build_static_graph(node, [successor], name=name)

    Encoder.__name__ = Encoder.__qualname__ = f"_{short}_encoder"
    Decoder.__name__ = Decoder.__qualname__ = f"_{short}_decoder"
    Node.__name__ = Node.__qualname__ = f"_{short}_node"
    return Node, Decoder
