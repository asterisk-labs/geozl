# WebAssembly

GeoZL builds as a wasm64 ES module. It needs Emscripten 4.0.19 or newer and a
runtime with WebAssembly Memory64 support, such as Node.js 24 or newer.

```sh
make wasm
make wasm-test
```

The build writes `geozl.js`, `geozl.wasm`, `geozl_api.js` and its TypeScript
declarations to `core/build-wasm/wasm/`.

```js
import {createGeoZL, DType} from './geozl_api.js';

const geozl = await createGeoZL();
const frame = geozl.compress(bytes, {
  method: 'planar>zigzag>pfor',
  width: 256,
  dtype: DType.U16,
});
const decoded = geozl.decompress(frame, {maxOutputSize: bytes.length});
```

Inputs and outputs are raw little-endian bytes. Dtype and shape stay outside
the frame. Browser deployments must serve both generated files; pass `wasmUrl`
to `createGeoZL` when they do not share a directory.
