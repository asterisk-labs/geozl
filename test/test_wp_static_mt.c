#include "wp_static/train_wp_static.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#define NTHREADS 8
#define ROWS 200
#define COLS 160
#define REPEATS 60

static uint16_t tiles[NTHREADS][ROWS * COLS];
static int16_t ref_coeffs[NTHREADS][4];
static uint8_t ref_shift[NTHREADS];
static int16_t got_coeffs[NTHREADS][4];
static uint8_t got_shift[NTHREADS];

static void *worker(void *arg) {
  const long t = (long)arg;
  for (int i = 0; i < REPEATS; ++i)
    wp_static_train(got_coeffs[t], &got_shift[t], tiles[t], COLS,
                    ROWS * COLS, sizeof(uint16_t));
  return NULL;
}

// A plane plus a gradient plus a little noise. Enough structure that the fit
// beats planar on some tiles and not others, which is what makes a torn
// histogram show up as a different choice rather than the same default.
static void fill_tiles(void) {
  unsigned seed = 12345u;
  for (int t = 0; t < NTHREADS; ++t) {
    for (int k = 0; k < ROWS * COLS; ++k) {
      seed = seed * 1103515245u + 12345u;
      const int r = k / COLS, c = k % COLS;
      tiles[t][k] = (uint16_t)(1000 + 40 * t + 7 * r + 3 * c
                               + ((seed >> 16) & 0x1F));
    }
  }
}

int main(void) {
  pthread_t threads[NTHREADS];
  int failures = 0;

  fill_tiles();

  for (int t = 0; t < NTHREADS; ++t)
    wp_static_train(ref_coeffs[t], &ref_shift[t], tiles[t], COLS,
                    ROWS * COLS, sizeof(uint16_t));

  for (long t = 0; t < NTHREADS; ++t) {
    if (pthread_create(&threads[t], NULL, worker, (void *)t) != 0) {
      printf("test_wp_static_mt: pthread_create failed\n");
      return 1;
    }
  }
  for (int t = 0; t < NTHREADS; ++t)
    pthread_join(threads[t], NULL);

  for (int t = 0; t < NTHREADS; ++t) {
    if (memcmp(ref_coeffs[t], got_coeffs[t], sizeof ref_coeffs[t]) != 0
        || ref_shift[t] != got_shift[t]) {
      printf("  FAIL tile %d  alone {%d,%d,%d,%d}>>%u  threaded {%d,%d,%d,%d}>>%u\n",
             t, ref_coeffs[t][0], ref_coeffs[t][1], ref_coeffs[t][2],
             ref_coeffs[t][3], ref_shift[t], got_coeffs[t][0], got_coeffs[t][1],
             got_coeffs[t][2], got_coeffs[t][3], got_shift[t]);
      ++failures;
    }
  }

  if (failures) {
    printf("test_wp_static_mt: %d of %d tiles diverged under %d threads\n",
           failures, NTHREADS, NTHREADS);
    return 1;
  }
  printf("test_wp_static_mt: ok\n");
  return 0;
}
