#ifndef GEOZL_CODECS_NODATA_WIRE_H
#define GEOZL_CODECS_NODATA_WIRE_H

// Wire header byte 0. One code today. The byte exists so a reader that meets a
// future one fails cleanly instead of misreading the pattern behind it.
#define GEOZL_NODATA_WIRE_RESTORE 1

#endif // GEOZL_CODECS_NODATA_WIRE_H
