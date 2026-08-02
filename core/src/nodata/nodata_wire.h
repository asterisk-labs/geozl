#ifndef GEOZL_CODECS_NODATA_WIRE_H
#define GEOZL_CODECS_NODATA_WIRE_H

// Wire header byte 0, which says how much of the pair of streams is really
// there. A reader must reject a code it does not know rather than interpret
// the bytes behind it.
//
//   1 restore    header 1 + eltWidth      values nbElts   mask nbElts
//   2 all valid  header 1                 values nbElts   mask empty
//   3 all hole   header 1 + eltWidth + 8  values empty    mask empty
//
// All hole drops both streams, so the sample count follows the pattern in the
// header as a little-endian uint64.
#define GEOZL_NODATA_WIRE_RESTORE 1
#define GEOZL_NODATA_WIRE_ALL_VALID 2
#define GEOZL_NODATA_WIRE_ALL_HOLE 3

#endif // GEOZL_CODECS_NODATA_WIRE_H
