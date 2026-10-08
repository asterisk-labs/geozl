export declare const DType: {
  readonly U8: 0;
  readonly U16: 1;
  readonly U32: 2;
  readonly U64: 3;
  readonly I8: 4;
  readonly I16: 5;
  readonly I32: 6;
  readonly I64: 7;
  readonly F16: 8;
  readonly F32: 9;
  readonly F64: 10;
};

export declare const NoData: {
  readonly NONE: 0;
  readonly NAN: 1;
  readonly VALUE: 2;
};

export type DTypeValue = (typeof DType)[keyof typeof DType];
export type NoDataValue = (typeof NoData)[keyof typeof NoData];

export interface CompressOptions {
  method: string;
  width: number;
  planes?: number;
  dtype: DTypeValue;
  error?: string | null;
  nodataMode?: NoDataValue;
  nodataBits?: number | bigint;
}

export interface DecompressOptions {
  verify?: boolean;
  maxOutputSize?: number | null;
}

export interface GeoZL {
  compress(data: Uint8Array, options: CompressOptions): Uint8Array;
  decompress(frame: Uint8Array, options?: DecompressOptions): Uint8Array;
}

export interface GeoZLOptions {
  wasmUrl?: string | URL;
  locateFile?: (path: string, prefix?: string) => string | URL;
  [option: string]: unknown;
}

export declare function createGeoZL(options?: GeoZLOptions): Promise<GeoZL>;
