// The entry point libFuzzer's main calls. libFuzzer ships no C header for it,
// and declaring it once keeps -Wmissing-prototypes on for every harness.

#ifndef GEOZL_FUZZ_FUZZER_H
#define GEOZL_FUZZ_FUZZER_H

#include <stddef.h>
#include <stdint.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

#endif // GEOZL_FUZZ_FUZZER_H
