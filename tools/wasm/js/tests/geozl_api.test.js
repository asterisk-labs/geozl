import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {describe, it} from 'node:test';

import {createGeoZL, DType, NoData} from '../geozl_api.js';

describe('GeoZL wasm64', () => {
  it('roundtrips a uint16 raster', async () => {
    const geozl = await createGeoZL();
    const data = new Uint8Array(64 * 64 * 2);
    const view = new DataView(data.buffer);
    for (let y = 0; y < 64; y++) {
      for (let x = 0; x < 64; x++) view.setUint16((y * 64 + x) * 2, 2000 + 8 * y + 5 * x, true);
    }
    const frame = geozl.compress(data, {
      method: 'planar>zigzag>pfor',
      width: 64,
      dtype: DType.U16,
    });
    assert.ok(frame.length < data.length);
    assert.deepEqual(geozl.decompress(frame), data);
  });

  it('checks geometry and output limits', async () => {
    const geozl = await createGeoZL();
    assert.throws(
      () => geozl.compress(new Uint8Array(10), {method: 'id>zstd', width: 4, dtype: DType.U16}),
      /width and planes/,
    );
    const data = new Uint8Array(64);
    const frame = geozl.compress(data, {method: 'id>zstd', width: 8, dtype: DType.U8});
    assert.throws(() => geozl.decompress(frame, {maxOutputSize: 63}), /maxOutputSize/);
  });

  it('preserves NaN and applies a bounded error', async () => {
    const geozl = await createGeoZL();
    const data = new Uint8Array(32 * 32 * 4);
    const src = new DataView(data.buffer);
    for (let i = 0; i < 32 * 32; i++) src.setFloat32(i * 4, i % 97 === 0 ? NaN : i / 7, true);
    const frame = geozl.compress(data, {
      method: 'planar>zigzag>transpose>zstd',
      width: 32,
      dtype: DType.F32,
      error: 'LINEAR:MAX_ERROR=0.5',
      nodataMode: NoData.NAN,
    });
    const back = new DataView(geozl.decompress(frame).buffer);
    for (let i = 0; i < 32 * 32; i++) {
      const expected = src.getFloat32(i * 4, true);
      const actual = back.getFloat32(i * 4, true);
      if (Number.isNaN(expected)) assert.ok(Number.isNaN(actual));
      else assert.ok(Math.abs(expected - actual) <= 0.5, `sample ${i}`);
    }
  });

  it('exports stable dtype and nodata codes', () => {
    assert.ok(Object.isFrozen(DType));
    assert.ok(Object.isFrozen(NoData));
    assert.equal(DType.F64, 10);
    assert.equal(NoData.VALUE, 2);
  });

  it('decodes released frames', async () => {
    const dir = process.env.GEOZL_GOLDEN_DIR;
    assert.ok(dir, 'GEOZL_GOLDEN_DIR is required');
    const geozl = await createGeoZL();
    const manifest = JSON.parse(readFileSync(`${dir}/manifest.json`, 'utf8'));
    for (const entry of manifest.frames) {
      const frame = readFileSync(`${dir}/frames/${entry.file}`);
      const decoded = geozl.decompress(frame);
      const digest = createHash('sha256').update(decoded).digest('hex');
      assert.equal(digest, entry.sha256_decoded, entry.file);
    }
  });
});
