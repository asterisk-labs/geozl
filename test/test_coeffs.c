#include "geozl/coeffs.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                 \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static uint8_t blob[GEOZL_COEFFS_MAX_BYTES + 64];
static char err[128];

static size_t pack2(const int32_t *a, uint32_t na, const int32_t *b,
                    uint32_t nb) {
  const int32_t *vecs[2] = {a, b};
  uint32_t counts[2] = {na, nb};
  return geozl_coeffs_pack(blob, sizeof blob, vecs, counts, b ? 2 : 1, err,
                           sizeof err);
}

static void test_round_trip(void) {
  const int32_t a[8] = {7, -3, 0, 11, 42, -99, INT32_MIN, INT32_MAX};
  const int32_t b[8] = {5, 8, -13, 21, -34, 55, -89, 144};
  const size_t n = pack2(a, 8, b, 8);
  CHECK(n == 6 + (4 + 32) + (4 + 32));

  size_t vecs = 0, values = 0;
  CHECK(geozl_coeffs_parse(blob, n, NULL, 0, NULL, 0, &vecs, &values) == 0);
  CHECK(vecs == 2);
  CHECK(values == 16);

  int32_t back[16];
  uint32_t counts[2];
  CHECK(geozl_coeffs_parse(blob, n, back, 16, counts, 2, NULL, NULL) == 0);
  CHECK(counts[0] == 8 && counts[1] == 8);
  CHECK(memcmp(back, a, sizeof a) == 0);
  CHECK(memcmp(back + 8, b, sizeof b) == 0);
}

static void test_single_value(void) {
  const int32_t s = 37;
  const size_t n = pack2(&s, 1, NULL, 0);
  CHECK(n == 6 + 4 + 4);
  int32_t back = 0;
  uint32_t counts[1];
  CHECK(geozl_coeffs_parse(blob, n, &back, 1, counts, 1, NULL, NULL) == 0);
  CHECK(counts[0] == 1);
  CHECK(back == s);
}

static void test_extremes(void) {
  const int32_t edge[5] = {INT32_MIN, -1, 0, 1, INT32_MAX};
  const size_t n = pack2(edge, 5, NULL, 0);
  CHECK(n != 0);
  int32_t back[5];
  CHECK(geozl_coeffs_parse(blob, n, back, 5, NULL, 0, NULL, NULL) == 0);
  CHECK(memcmp(back, edge, sizeof edge) == 0);
}

static void test_invalid_shapes(void) {
  const int32_t v[2] = {1, 2};
  const int32_t *vecs[2] = {v, v};
  uint32_t counts[2] = {2, 2};

  CHECK(geozl_coeffs_pack(blob, sizeof blob, vecs, counts, 0, err,
                          sizeof err) == 0);
  CHECK(err[0] != '\0');
  counts[1] = 0;
  CHECK(geozl_coeffs_pack(blob, sizeof blob, vecs, counts, 2, err,
                          sizeof err) == 0);
  counts[1] = 2;
  vecs[1] = NULL;
  CHECK(geozl_coeffs_pack(blob, sizeof blob, vecs, counts, 2, err,
                          sizeof err) == 0);
  vecs[1] = v;
  CHECK(geozl_coeffs_pack(blob, 6 + 4 + 8 + 4 + 8 - 1, vecs, counts, 2, err,
                          sizeof err) == 0);
}

static void test_size_limit(void) {
  static int32_t big[2600];
  for (size_t i = 0; i < 2600; ++i)
    big[i] = (int32_t)i;
  CHECK(pack2(big, 1248, big, 1248) == 9998);
  CHECK(pack2(big, 1249, big, 1249) == 0);
  CHECK(strstr(err, "10006") != NULL);
  CHECK(pack2(big, 2497, NULL, 0) == 9998);
  CHECK(pack2(big, 2498, NULL, 0) == 0);
}

static void test_foreign_blobs(void) {
  size_t vecs = 7, values = 9;
  const int32_t v = 1;
  const size_t n = pack2(&v, 1, NULL, 0);

  CHECK(geozl_coeffs_parse(NULL, 0, NULL, 0, NULL, 0, &vecs, &values) == 1);
  CHECK(vecs == 0 && values == 0);
  // Inputs shorter than the magic are foreign blobs.
  CHECK(geozl_coeffs_parse(blob, 3, NULL, 0, NULL, 0, &vecs, NULL) == 1);
  uint8_t foreign[32] = {'h', 'e', 'l', 'l', 'o'};
  CHECK(geozl_coeffs_parse(foreign, sizeof foreign, NULL, 0, NULL, 0, &vecs,
                           NULL) == 1);
  CHECK(n != 0);
}

static void test_invalid_blobs(void) {
  const int32_t v[3] = {1, 2, 3};
  const size_t n = pack2(v, 3, NULL, 0);
  CHECK(n == 6 + 4 + 12);

  uint8_t copy[64];
  size_t vecs = 0;

  // The first count is small enough to change through its low byte.
  struct {
    const char *what;
    size_t extra;
    size_t at;
    uint8_t to;
  } broken[] = {
      {"unknown version", 0, 4, 2},
      {"no vectors", 0, 5, 0},
      {"a value count of zero", 0, 6, 0},
      {"a count that runs past the comment", 0, 7, 0xFF},
      {"one trailing byte", 1, 0, 'G'},
  };
  for (size_t i = 0; i < sizeof broken / sizeof *broken; ++i) {
    memcpy(copy, blob, n);
    copy[broken[i].at] = broken[i].to;
    copy[n] = 0;
    const size_t sz = n + broken[i].extra;
    if (geozl_coeffs_parse(copy, sz, NULL, 0, NULL, 0, &vecs, NULL) != -1) {
      printf("  FAIL %s: %s did not read as corrupt\n", __FILE__,
             broken[i].what);
      ++failures;
    }
  }
  CHECK(geozl_coeffs_parse(blob, n - 1, NULL, 0, NULL, 0, &vecs, NULL) == -1);

  // A complete magic with a truncated header is invalid.
  CHECK(geozl_coeffs_parse(blob, 4, NULL, 0, NULL, 0, &vecs, NULL) == -1);
  CHECK(geozl_coeffs_parse(blob, 5, NULL, 0, NULL, 0, &vecs, NULL) == -1);
  CHECK(geozl_coeffs_parse(blob, 3, NULL, 0, NULL, 0, &vecs, NULL) == 1);

  static uint8_t huge[GEOZL_COEFFS_MAX_BYTES + 8];
  memcpy(huge, blob, n);
  CHECK(geozl_coeffs_parse(huge, sizeof huge, NULL, 0, NULL, 0, &vecs, NULL) ==
        -1);

  int32_t one = 7;
  CHECK(geozl_coeffs_parse(blob, n, &one, 1, NULL, 0, NULL, NULL) == -2);
  CHECK(one == 7);
  uint32_t counts[1];
  const int32_t pair[2] = {1, 2};
  const size_t m = pack2(pair, 1, pair, 1);
  CHECK(m != 0);
  CHECK(geozl_coeffs_parse(blob, m, NULL, 0, counts, 1, NULL, NULL) == -2);
}

static void test_vector_ceiling(void) {
  static const int32_t one = 1;
  static const int32_t *vecs[GEOZL_COEFFS_MAX_VECS + 1];
  static uint32_t counts[GEOZL_COEFFS_MAX_VECS + 1];
  for (size_t i = 0; i <= GEOZL_COEFFS_MAX_VECS; ++i) {
    vecs[i] = &one;
    counts[i] = 1;
  }
  CHECK(geozl_coeffs_pack(blob, sizeof blob, vecs, counts,
                          GEOZL_COEFFS_MAX_VECS, err, sizeof err) ==
        6 + GEOZL_COEFFS_MAX_VECS * (4 + 4));
  CHECK(geozl_coeffs_size(counts, GEOZL_COEFFS_MAX_VECS + 1) == 0);
  CHECK(geozl_coeffs_pack(blob, sizeof blob, vecs, counts,
                          GEOZL_COEFFS_MAX_VECS + 1, err, sizeof err) == 0);
}

static void test_count_cut_in_half(void) {
  const int32_t v[2] = {5, 6};
  const size_t n = pack2(v, 2, NULL, 0);
  size_t vecs = 0;
  // Eight bytes leaves the header and half of the first count.
  CHECK(n > 8);
  CHECK(geozl_coeffs_parse(blob, 8, NULL, 0, NULL, 0, &vecs, NULL) == -1);
}

static void test_wire_bytes(void) {
  static const uint8_t kOne[] = {
      0x47, 0x5a, 0x43, 0x31,              // "GZC1"
      0x01,                                // version
      0x01,                                // one vector
      0x01, 0x00, 0x00, 0x00,              // holding one value
      0x25, 0x00, 0x00, 0x00,              // 37
  };
  const int32_t s = 37;
  CHECK(pack2(&s, 1, NULL, 0) == sizeof kOne);
  CHECK(memcmp(blob, kOne, sizeof kOne) == 0);

  static const uint8_t kTwo[] = {
      0x47, 0x5a, 0x43, 0x31, 0x01, 0x02,
      0x02, 0x00, 0x00, 0x00,
      0x07, 0x00, 0x00, 0x00,              // 7
      0xd6, 0xff, 0xff, 0xff,              // -42
      0x02, 0x00, 0x00, 0x00,
      0xe8, 0x03, 0x00, 0x00,              // 1000
      0x30, 0xf8, 0xff, 0xff,              // -2000
  };
  const int32_t left[2] = {7, -42};
  const int32_t right[2] = {1000, -2000};
  CHECK(pack2(left, 2, right, 2) == sizeof kTwo);
  CHECK(memcmp(blob, kTwo, sizeof kTwo) == 0);

  // Verify two's-complement encoding for negative values.
  static const uint8_t kNeg[] = {
      0x47, 0x5a, 0x43, 0x31, 0x01, 0x01,
      0x02, 0x00, 0x00, 0x00,
      0xfe, 0xff, 0xff, 0xff,              // -2
      0x00, 0x00, 0x00, 0x80,              // INT32_MIN
  };
  const int32_t neg[2] = {-2, INT32_MIN};
  CHECK(pack2(neg, 2, NULL, 0) == sizeof kNeg);
  CHECK(memcmp(blob, kNeg, sizeof kNeg) == 0);
}

static void test_size(void) {
  uint32_t counts[3] = {1, 5, 9};
  const int32_t v[9] = {0};
  const int32_t *vecs[3] = {v, v, v};
  const size_t want = geozl_coeffs_size(counts, 3);
  CHECK(want == 6 + (4 + 4) + (4 + 20) + (4 + 36));
  CHECK(geozl_coeffs_pack(blob, sizeof blob, vecs, counts, 3, err,
                          sizeof err) == want);
  counts[1] = 0;
  CHECK(geozl_coeffs_size(counts, 3) == 0);
  CHECK(geozl_coeffs_size(counts, 0) == 0);
}

int main(void) {
  test_round_trip();
  test_single_value();
  test_extremes();
  test_invalid_shapes();
  test_size_limit();
  test_foreign_blobs();
  test_invalid_blobs();
  test_vector_ceiling();
  test_count_cut_in_half();
  test_wire_bytes();
  test_size();

  if (failures) {
    printf("test_coeffs: %d failed\n", failures);
    return 1;
  }
  printf("test_coeffs: ok\n");
  return 0;
}
