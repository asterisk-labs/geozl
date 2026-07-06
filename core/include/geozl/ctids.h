#ifndef GEOZL_CTIDS_H
#define GEOZL_CTIDS_H

// Every geozl codec id in one place. A CTid is the only identity a frame
// carries, names never travel, so this table is the interop contract.
//
// The bands separate the two reconstruction contracts, so a reader can tell a
// lossless codec from a lossy one from the id alone.
//   lossless, decode(encode(x)) == x        0x72D700 to 0x72D77F
//   lossy, reconstruction within a bound     0x72D780 to 0x72D7FF
// The whole block sits high enough to stay clear of OpenZL's standard node ids.

// Lossless
#define GEOZL_CTID_DELTA_W      0x72D701
#define GEOZL_CTID_DELTA_N      0x72D702
#define GEOZL_CTID_PLANAR       0x72D703
#define GEOZL_CTID_DEINTERLEAVE 0x72D704
#define GEOZL_CTID_MED          0x72D705
#define GEOZL_CTID_AVERAGE      0x72D706
#define GEOZL_CTID_WP_STATIC    0x72D707

// Lossy
#define GEOZL_CTID_QUANT_LINEAR 0x72D780

#endif // GEOZL_CTIDS_H
