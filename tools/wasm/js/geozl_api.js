export const DType = Object.freeze({
  U8: 0,
  U16: 1,
  U32: 2,
  U64: 3,
  I8: 4,
  I16: 5,
  I32: 6,
  I64: 7,
  F16: 8,
  F32: 9,
  F64: 10,
});

export const NoData = Object.freeze({NONE: 0, NAN: 1, VALUE: 2});

const DTYPE_WIDTH = Object.freeze([1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8]);
const ERROR_BYTES = 256;
const U64_MAX = (1n << 64n) - 1n;

function checkPositiveInt(value, name, max = 0xffffffff) {
  if (!Number.isInteger(value) || value < 1 || value > max) {
    throw new Error(`${name} must be an integer from 1 to ${max}`);
  }
}

function toU64(value, name) {
  if (typeof value === 'number') {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`${name} must be a non-negative safe integer or bigint`);
    }
    return BigInt(value);
  }
  if (typeof value !== 'bigint' || value < 0n || value > U64_MAX) {
    throw new Error(`${name} must fit in uint64`);
  }
  return value;
}

function loadSize(mod, ptr) {
  const value = mod.HEAPU64[Math.floor(ptr / 8)];
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error('wasm result is too large for JavaScript');
  }
  return Number(value);
}

function createMemory(mod) {
  const wasm64 = (value) => BigInt(value);
  const malloc = (size) => {
    const ptr = Number(mod._geozl_wasm_malloc(wasm64(size)));
    if (!ptr) throw new Error('wasm malloc failed');
    return ptr;
  };
  const free = (ptr) => {
    if (ptr) mod._geozl_wasm_free(wasm64(ptr));
  };
  const writeBytes = (bytes) => {
    const ptr = malloc(bytes.length);
    mod.HEAPU8.set(bytes, ptr);
    return ptr;
  };
  const writeString = (value) => {
    const bytes = new TextEncoder().encode(`${value}\0`);
    return writeBytes(bytes);
  };
  return {wasm64, malloc, free, writeBytes, writeString};
}

function nativeError(mod, ptr, operation, code) {
  const detail = mod.UTF8ToString(ptr);
  const error = new Error(`${operation} failed${detail ? `: ${detail}` : ''} (ZL error ${code})`);
  error.code = code;
  return error;
}

async function loadModule(options) {
  let factory;
  try {
    ({default: factory} = await import('./geozl.js'));
  } catch (cause) {
    throw new Error('geozl.js is missing; run `make wasm` first', {cause});
  }
  const {wasmUrl = new URL('./geozl.wasm', import.meta.url).href, locateFile, ...moduleOptions} = options;
  return factory({
    ...moduleOptions,
    locateFile: locateFile ?? ((path, prefix = '') => (path.endsWith('.wasm') ? wasmUrl : `${prefix}${path}`)),
  });
}

export async function createGeoZL(options = {}) {
  const mod = await loadModule(options);
  const mem = createMemory(mod);

  return Object.freeze({
    compress(data, options = {}) {
      if (!(data instanceof Uint8Array)) throw new Error('compress expects Uint8Array');
      const {
        method,
        width,
        planes = 1,
        dtype,
        error = null,
        nodataMode = NoData.NONE,
        nodataBits = 0n,
      } = options;
      if (typeof method !== 'string' || method.length === 0) {
        throw new Error('method must be a non-empty string');
      }
      checkPositiveInt(width, 'width');
      checkPositiveInt(planes, 'planes');
      if (!Number.isInteger(dtype) || dtype < DType.U8 || dtype > DType.F64) {
        throw new Error('dtype must be a DType value');
      }
      if (error !== null && typeof error !== 'string') {
        throw new Error('error must be a recipe string or null');
      }
      if (!Object.values(NoData).includes(nodataMode)) {
        throw new Error('nodataMode must be a NoData value');
      }

      const eltWidth = DTYPE_WIDTH[dtype];
      if (data.length === 0 || data.length % eltWidth !== 0) {
        throw new Error(`data length must be a non-zero multiple of ${eltWidth}`);
      }
      const numElts = data.length / eltWidth;
      if (numElts % planes !== 0 || (numElts / planes) % width !== 0) {
        throw new Error('data length does not match width and planes');
      }

      const capacity = 1024 + data.length + Math.floor(data.length / 2);
      let methodPtr = 0;
      let errorPtr = 0;
      let srcPtr = 0;
      let dstPtr = 0;
      let outSizePtr = 0;
      let errorCtxPtr = 0;
      try {
        methodPtr = mem.writeString(method);
        if (error !== null && error.length !== 0) errorPtr = mem.writeString(error);
        srcPtr = mem.writeBytes(data);
        dstPtr = mem.malloc(capacity);
        outSizePtr = mem.malloc(8);
        errorCtxPtr = mem.malloc(ERROR_BYTES);
        const code = mod._geozl_wasm_compress(
          mem.wasm64(methodPtr),
          width,
          planes,
          mem.wasm64(errorPtr),
          dtype,
          nodataMode,
          toU64(nodataBits, 'nodataBits'),
          mem.wasm64(srcPtr),
          mem.wasm64(numElts),
          mem.wasm64(eltWidth),
          mem.wasm64(dstPtr),
          mem.wasm64(capacity),
          mem.wasm64(outSizePtr),
          mem.wasm64(errorCtxPtr),
          mem.wasm64(ERROR_BYTES),
        );
        if (code !== 0) throw nativeError(mod, errorCtxPtr, 'compress', code);
        const outSize = loadSize(mod, outSizePtr);
        if (outSize > capacity) throw new Error('compress returned an invalid size');
        return mod.HEAPU8.slice(dstPtr, dstPtr + outSize);
      } finally {
        for (const ptr of [methodPtr, errorPtr, srcPtr, dstPtr, outSizePtr, errorCtxPtr]) mem.free(ptr);
      }
    },

    decompress(frame, options = {}) {
      if (!(frame instanceof Uint8Array) || frame.length === 0) {
        throw new Error('decompress expects a non-empty Uint8Array');
      }
      const {verify = true, maxOutputSize = null} = options;
      if (typeof verify !== 'boolean') throw new Error('verify must be boolean');
      if (
        maxOutputSize !== null &&
        (!Number.isSafeInteger(maxOutputSize) || maxOutputSize < 0)
      ) {
        throw new Error('maxOutputSize must be a non-negative safe integer or null');
      }

      let framePtr = 0;
      let dstPtr = 0;
      let outSizePtr = 0;
      let errorCtxPtr = 0;
      try {
        framePtr = mem.writeBytes(frame);
        const rawSize = mod._geozl_wasm_frame_dsize(mem.wasm64(framePtr), mem.wasm64(frame.length));
        if (rawSize === 0n) throw new Error('decompress failed: unreadable frame');
        if (rawSize > BigInt(Number.MAX_SAFE_INTEGER)) {
          throw new Error('frame output is too large for JavaScript');
        }
        const size = Number(rawSize);
        if (maxOutputSize !== null && size > maxOutputSize) {
          throw new Error(`frame declares ${size} bytes, above maxOutputSize ${maxOutputSize}`);
        }
        dstPtr = mem.malloc(size);
        outSizePtr = mem.malloc(8);
        errorCtxPtr = mem.malloc(ERROR_BYTES);
        const code = mod._geozl_wasm_decompress(
          mem.wasm64(framePtr),
          mem.wasm64(frame.length),
          mem.wasm64(dstPtr),
          mem.wasm64(size),
          mem.wasm64(outSizePtr),
          verify ? 1 : 0,
          mem.wasm64(errorCtxPtr),
          mem.wasm64(ERROR_BYTES),
        );
        if (code !== 0) throw nativeError(mod, errorCtxPtr, 'decompress', code);
        const outSize = loadSize(mod, outSizePtr);
        if (outSize > size) throw new Error('decompress returned an invalid size');
        return mod.HEAPU8.slice(dstPtr, dstPtr + outSize);
      } finally {
        for (const ptr of [framePtr, dstPtr, outSizePtr, errorCtxPtr]) mem.free(ptr);
      }
    },
  });
}
