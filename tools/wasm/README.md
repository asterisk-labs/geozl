# WebAssembly

GeoZL builds as a wasm64 ES module. It needs a runtime with WebAssembly
Memory64 support, such as Node.js 24 or newer.

```sh
npm install @asterisk-labs/geozl
```

```js
import {createGeoZL, DType} from '@asterisk-labs/geozl';

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

Building it needs Emscripten 4.0.19 or newer. The build writes the package,
`geozl.js`, `geozl.wasm`, `geozl_api.js` and its TypeScript declarations, to
`core/build-wasm/wasm/`.

```sh
make wasm
make wasm-test
```
