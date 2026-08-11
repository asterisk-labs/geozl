// The dtype tables live in four places, geozl/dtype.h and one copy per lossy
// family. They carry the same wire codes so a codec folder lifts out whole.
// Nothing held them to it before this.

#include "geozl/dtype.h"

#include "quant_linear/quant_linear_dtype.h"
#include "quant_log/quant_log_dtype.h"
#include "quant_sqrt/quant_sqrt_dtype.h"

#include <stdio.h>

static int failures = 0;

static void eq_size(const char *what, int dtype, size_t a, size_t b,
                    const char *an, const char *bn) {
  if (a != b) {
    printf("  dtype %2d %-12s %s=%zu but %s=%zu\n", dtype, what, an, a, bn, b);
    ++failures;
  }
}

// Bit for bit. A copy merely close to another has already drifted.
static void eq_double(const char *what, int dtype, double a, double b,
                      const char *an, const char *bn) {
  if (!(a == b)) {
    printf("  dtype %2d %-12s %s=%.17g but %s=%.17g\n", dtype, what, an, a, bn,
           b);
    ++failures;
  }
}

static void eq_int(const char *what, int dtype, int a, int b, const char *an,
                   const char *bn) {
  if (a != b) {
    printf("  dtype %2d %-12s %s=%d but %s=%d\n", dtype, what, an, a, bn, b);
    ++failures;
  }
}

int main(void) {
  printf("dtype tables agree across the four copies\n");

  // Same numbers, or a header written by one family reads as another type in
  // the next.
  eq_int("enum U8", 0, (int)GEOZL_DT_U8, (int)QL_U8, "geozl", "linear");
  eq_int("enum F64", 10, (int)GEOZL_DT_F64, (int)QL_F64, "geozl", "linear");
  eq_int("enum F64", 10, (int)QL_F64, (int)QLOG_F64, "linear", "log");
  eq_int("enum F64", 10, (int)QL_F64, (int)QSQ_F64, "linear", "sqrt");
  eq_int("last int", 0, (int)GEOZL_DT_LAST_INT, (int)QL_LAST_INT, "geozl",
         "linear");
  eq_int("last int", 0, (int)QL_LAST_INT, (int)QLOG_LAST_INT, "linear", "log");
  eq_int("last int", 0, (int)QL_LAST_INT, (int)QSQ_LAST_INT, "linear", "sqrt");

  for (int d = GEOZL_DT_U8; d <= GEOZL_DT_F64; ++d) {
    eq_int("dtype_ok", d, GEOZL_DT_OK(d) ? 1 : 0, QL_DTYPE_OK(d) ? 1 : 0,
           "geozl", "linear");
    eq_int("dtype_ok", d, GEOZL_DT_OK(d) ? 1 : 0, QLOG_DTYPE_OK(d) ? 1 : 0,
           "geozl", "log");
    eq_int("dtype_ok", d, GEOZL_DT_OK(d) ? 1 : 0, QSQ_DTYPE_OK(d) ? 1 : 0,
           "geozl", "sqrt");

    // A disagreement here is an out of bounds read, not a wrong number.
    const size_t w = geozl_dtype_width(d);
    eq_size("width", d, w, quant_linear_width(d), "geozl", "linear");
    eq_size("width", d, w, quant_log_width(d), "geozl", "log");
    eq_size("width", d, w, quant_sqrt_width(d), "geozl", "sqrt");

    // Both ends read the output range from these.
    const double hi = quant_linear_value_hi(d);
    eq_double("value_hi", d, hi, quant_log_value_hi(d), "linear", "log");
    eq_double("value_hi", d, hi, quant_sqrt_value_hi(d), "linear", "sqrt");

    // quant_log only reaches its floor on the index path, which an integer
    // never takes, so its integer arm returns the f64 floor by contract.
    const double lo = quant_linear_value_lo(d);
    eq_double("value_lo", d, lo, quant_sqrt_value_lo(d), "linear", "sqrt");
    if (d >= GEOZL_DT_F16)
      eq_double("value_lo", d, lo, quant_log_value_lo(d), "linear", "log");
  }

  if (failures != 0) {
    printf("FAIL, %d disagreement(s)\n", failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
