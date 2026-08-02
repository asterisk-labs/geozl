# geozl build. Common targets, see `make help` for the rest.
#
#   make            build and install the Python binding, then smoke load it
#   make test       run the C tests then the Python suite
#   make test-c     run the C tests only, no cmake or OpenZL needed
#   make test-san   run the suite under ASan and UBSan
#   make fuzz       build and run both fuzzers, output under fuzz/out
#   make test-exhaustive  walk every float32 through every recipe
#   make clean      remove all build output and generated files
#
# Vendors OpenZL as a submodule, fetched on first build.

PYTHON ?= python
PREFIX ?= /usr/local
BUILD  ?= Release
GEN    ?= Ninja
FULL   ?= ON
SAN    ?= OFF

# Seconds per fuzz target, and libFuzzer worker processes. FUZZ_JOBS=0 keeps it
# in one process, which is what you want when reading the log.
FUZZ_TIME ?= 60
FUZZ_JOBS ?= 0

CORE       := core
FUZZ_OUT   := fuzz/out
OPENZL     := extern/openzl
PY_DIR     := bindings/python
PY_LIB_DIR := $(PY_DIR)/geozl/_lib
UNAME      := $(shell uname -s)

# The sanitized build gets its own tree, and the test run preloads the ASan
# runtime because the interpreter that dlopens the kernels lib is not instrumented.
ifeq ($(SAN),ON)
  BUILD_DIR := core/build-san
  ifeq ($(UNAME),Darwin)
    SAN_RT := $(shell $(CC) -print-runtime-dir)/libclang_rt.asan_osx_dynamic.dylib
    SAN_PRELOAD := DYLD_INSERT_LIBRARIES=$(SAN_RT)
  else
    # libstdc++ too, so ASan can resolve __cxa_throw when a codec callback raises
    # a Python exception back through the openzl C++ layer.
    SAN_RT := $(shell $(CC) -print-file-name=libasan.so):$(shell $(CC) -print-file-name=libstdc++.so)
    SAN_PRELOAD := LD_PRELOAD=$(SAN_RT)
  endif
  # macOS strips DYLD_* from spawned children, so pass the path through a plain
  # var the test can reinject as the preload.
  SAN_ENV := $(SAN_PRELOAD) GEOZL_ASAN_RT=$(SAN_RT) \
             ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 \
             UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
else
  BUILD_DIR := core/build
endif

ifeq ($(UNAME),Darwin)
  KERNELS := libgeozl_kernels.dylib
  FULLLIB := libgeozl.dylib
else ifeq ($(OS),Windows_NT)
  KERNELS := geozl_kernels.dll
  FULLLIB := geozl.dll
else
  KERNELS := libgeozl_kernels.so
  FULLLIB := libgeozl.so
endif

CMAKE_FLAGS ?=
CMAKE_OPTS  := -G $(GEN) -DCMAKE_BUILD_TYPE=$(BUILD) \
               -DGEOZL_BUILD_FULL=$(FULL) -DGEOZL_BUILD_KERNELS_SHARED=ON \
               -DGEOZL_SANITIZE=$(SAN) $(CMAKE_FLAGS)

# Standalone C tests. The kernels carry no OpenZL dependency, so these compile
# and run straight from source without configuring cmake, which keeps the first
# half of `make test` useful on a tree whose submodule is not fetched yet.
CTEST_DIR     := $(BUILD_DIR)/ctest
# The exhaustive walk takes minutes, so it is not part of test-c. See
# test-exhaustive below.
CTEST_SRCS    := $(filter-out test/test_quant_exhaustive.c \
                              test/test_quant_log_exhaustive.c,\
                   $(wildcard test/test_*.c))
CTEST_BINS    := $(patsubst test/%.c,$(CTEST_DIR)/%,$(CTEST_SRCS))
# scan.h dispatches through geozl_simd_now, so anything including it needs
# simd.c at link time. Listed by hand, like train_wp_static.c, because neither
# matches the kernel glob.
CTEST_KERNELS := $(wildcard $(CORE)/src/*/encode_*_kernel.c) \
                 $(wildcard $(CORE)/src/*/decode_*_kernel.c) \
                 $(CORE)/src/wp_static/train_wp_static.c \
                 $(CORE)/src/quant/quant_spec.c \
                 $(CORE)/src/quant_linear/quant_linear_spec.c \
                 $(CORE)/src/quant_sqrt/quant_sqrt_spec.c \
                 $(CORE)/src/quant_sqrt/quant_sqrt_fit.c \
                 $(CORE)/src/quant_log/quant_log_spec.c \
                 $(CORE)/src/common/simd.c
# include/ too: the quant kernels take their parameter block from
# geozl/quant_params.h, which is public because geozl_node_quant is.
# -ffp-contract=off for the same reason core/CMakeLists.txt sets it, the
# reconstruction has to be the same bits on every machine that reads a frame.
CTEST_CFLAGS  := -std=c11 -O1 -g -ffp-contract=off -Wall -Wextra \
                 -I$(CORE)/include -I$(CORE)/src

# No SAN_ENV here, the test binaries are instrumented themselves. Leak
# detection is left on, that is how a missing free in a kernel shows up.
ifeq ($(SAN),ON)
  # float-cast-overflow is not part of undefined in gcc, and a double that does
  # not fit the integer it is cast to is exactly how a forged parameter block
  # reaches undefined behaviour.
  CTEST_CFLAGS += -fsanitize=address,undefined,float-cast-overflow \
                  -fno-omit-frame-pointer
endif
ifeq ($(OS),Windows_NT)
  CTEST_LIBS := -lm
else
  CTEST_LIBS := -lm -lpthread
endif

.PHONY: all build configure lib python test test-c test-san fuzz fuzz-build \
        fuzz-report clean-fuzz test-exhaustive install submodules clean help

all: python

# A fresh clone has an empty submodule, fetch it so a bare make works.
$(OPENZL)/CMakeLists.txt:
	git submodule update --init --recursive

submodules:
	git submodule update --init --recursive

$(BUILD_DIR)/CMakeCache.txt: $(OPENZL)/CMakeLists.txt
	cmake -S $(CORE) -B $(BUILD_DIR) $(CMAKE_OPTS)

configure: $(OPENZL)/CMakeLists.txt
	cmake -S $(CORE) -B $(BUILD_DIR) $(CMAKE_OPTS)

build: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR)

# Stage the kernels lib next to the binding, cffi loads it from there.
lib: build
	@mkdir -p $(PY_LIB_DIR)
	@rm -f $(PY_LIB_DIR)/libgeozl_kernels* $(PY_LIB_DIR)/geozl_kernels*.dll
	@f=$$(find $(BUILD_DIR) \( -name 'libgeozl_kernels*.dylib' \
	        -o -name 'libgeozl_kernels*.so*' -o -name 'geozl_kernels*.dll' \) | head -1); \
	  [ -n "$$f" ] || { echo "no $(KERNELS) under $(BUILD_DIR)"; exit 1; }; \
	  cp -a "$$f" $(PY_LIB_DIR)/
	@rm -f $(PY_LIB_DIR)/libgeozl.dylib $(PY_LIB_DIR)/libgeozl.so* $(PY_LIB_DIR)/geozl.dll
	@g=$$(find $(BUILD_DIR) \( -name 'libgeozl.dylib' \
	        -o -name 'libgeozl.so*' -o -name 'geozl.dll' \) | head -1); \
	  if [ -n "$$g" ]; then cp -a "$$g" $(PY_LIB_DIR)/; \
	  else echo "note: no $(FULLLIB) (FULL=OFF), geozl.compress unavailable"; fi

python: lib
	@$(PYTHON) -c 'import numpy, cffi, openzl' 2>/dev/null \
	  || { echo "missing runtime deps, install: numpy cffi openzl"; exit 1; }
	$(PYTHON) -m pip install -e $(PY_DIR) -q
	$(SAN_ENV) $(PYTHON) -c "import geozl; print('geozl', geozl.__version__)"

# Every test/test_*.c links against all kernels, so a new test can call any of
# them without touching this file.
$(CTEST_DIR)/%: test/%.c $(CTEST_KERNELS)
	@mkdir -p $(CTEST_DIR)
	@$(CC) $(CTEST_CFLAGS) $^ $(CTEST_LIBS) -o $@

test-c: $(CTEST_BINS)
	@if [ -z "$(CTEST_BINS)" ]; then echo "no C tests under test/"; else \
	  for t in $(CTEST_BINS); do "$$t" || exit 1; done; fi

test: test-c python
	@$(PYTHON) -c 'import pytest' 2>/dev/null || { echo "pytest not installed"; exit 1; }
	@$(SAN_ENV) $(PYTHON) -m pytest -q $(PY_DIR); \
	  rc=$$?; if [ $$rc -eq 5 ]; then echo "no tests collected"; elif [ $$rc -ne 0 ]; then exit $$rc; fi

test-san:
	$(MAKE) test SAN=ON

# Apple Command Line Tools clang has no fuzzer runtime, so on macOS default to
# Homebrew LLVM. Override with CLANG=/path/to/clang.
ifeq ($(UNAME),Darwin)
  BREW_LLVM := $(shell brew --prefix llvm 2>/dev/null)/bin/clang
  CLANG ?= $(if $(wildcard $(BREW_LLVM)),$(BREW_LLVM),clang)
else
  CLANG ?= clang
endif

fuzz-build: $(OPENZL)/CMakeLists.txt python
	cmake -S $(CORE) -B core/build-fuzz -G $(GEN) -DCMAKE_BUILD_TYPE=$(BUILD) \
	      -DGEOZL_BUILD_FULL=ON -DGEOZL_SANITIZE=ON -DGEOZL_BUILD_FUZZERS=ON \
	      -DCMAKE_C_COMPILER=$(CLANG) -DCMAKE_CXX_COMPILER=$(CLANG)++
	cmake --build core/build-fuzz --target geozl_decode_fuzzer geozl_quant_fuzzer
	@$(PYTHON) fuzz/gen_corpus.py fuzz/corpus

# libFuzzer writes its per job logs to the working directory and its findings to
# artifact_prefix, so both runs happen from inside fuzz/out and nothing lands in
# the repo root. allocator_may_return_null lets an absurd allocation come back
# as null instead of aborting, which OpenZL handles: without it the decode run
# stops every few minutes on a forged size field rather than fuzzing.
fuzz: fuzz-build
	@mkdir -p $(FUZZ_OUT)/decode-work
	@echo "quant fuzzer, $(FUZZ_TIME)s"
	@cd $(FUZZ_OUT) && ASAN_OPTIONS=allocator_may_return_null=1 \
	  $(abspath core/build-fuzz)/geozl_quant_fuzzer \
	  -max_total_time=$(FUZZ_TIME) -jobs=$(FUZZ_JOBS) \
	  -artifact_prefix=$(abspath $(FUZZ_OUT))/ > quant.log 2>&1 || true
	@echo "decode fuzzer, $(FUZZ_TIME)s"
	@cd $(FUZZ_OUT) && ASAN_OPTIONS=allocator_may_return_null=1 \
	  $(abspath core/build-fuzz)/geozl_decode_fuzzer decode-work \
	  $(abspath fuzz/corpus) -max_total_time=$(FUZZ_TIME) -max_len=4096 \
	  -rss_limit_mb=0 -jobs=$(FUZZ_JOBS) \
	  -artifact_prefix=$(abspath $(FUZZ_OUT))/ > decode.log 2>&1 || true
	@$(MAKE) --no-print-directory fuzz-report

# One file worth reading or handing over. The findings are dumped as hex so a
# report says everything without a binary attached.
fuzz-report:
	@{ \
	  echo "geozl fuzz report"; \
	  echo "$(FUZZ_TIME)s per target, jobs $(FUZZ_JOBS)"; \
	  for t in quant decode; do \
	    echo; echo "== $$t =="; \
	    grep -hE 'INITED|DONE|Loaded . modules' \
	      $(FUZZ_OUT)/$$t.log 2>/dev/null | head -4 || true; \
	    grep -hB2 -A12 -E 'runtime error|ERROR:|SUMMARY:' \
	      $(FUZZ_OUT)/$$t.log 2>/dev/null | head -30 || true; \
	  done; \
	  echo; echo "== findings =="; \
	  found=$$(ls -1 $(FUZZ_OUT) 2>/dev/null | grep -E '^(crash|oom|leak|timeout)-'); \
	  if [ -z "$$found" ]; then echo "none"; else \
	    for f in $$found; do \
	      echo "--- $$f ($$(wc -c < $(FUZZ_OUT)/$$f) bytes)"; \
	      od -An -tx1 -v $(FUZZ_OUT)/$$f | head -16; \
	    done; \
	  fi; \
	} > $(FUZZ_OUT)/report.txt
	@cat $(FUZZ_OUT)/report.txt
	@echo; echo "full report in $(FUZZ_OUT)/report.txt"

# Every bit pattern a float32 can hold, through every recipe. Fuzzing says
# nobody found a counterexample, this says there is not one. The methodology is
# from Fallin and Burtscher, arXiv:2407.15037, who tested LC the same way.
# EXH_N below 2^32 strides across the space for a quick pass.
EXH_JOBS ?= 8
EXH_N    ?= 4294967296

$(CTEST_DIR)/test_quant_exhaustive: CTEST_CFLAGS += -O2
$(CTEST_DIR)/test_quant_log_exhaustive: CTEST_CFLAGS += -O2

# quant_log walks u8, u16, i16 and f16 whole as well as every normal float32, and
# takes no arguments because none of those spaces is worth striding across.
test-exhaustive: $(CTEST_DIR)/test_quant_exhaustive \
                 $(CTEST_DIR)/test_quant_log_exhaustive
	@$(CTEST_DIR)/test_quant_exhaustive $(EXH_JOBS) $(EXH_N)
	@$(CTEST_DIR)/test_quant_log_exhaustive

clean-fuzz:
	rm -rf $(FUZZ_OUT) fuzz/corpus core/build-fuzz
	rm -f crash-* leak-* timeout-* oom-* fuzz-*.log

install: build
	cmake --install $(BUILD_DIR) --prefix $(PREFIX)

clean: clean-fuzz
	rm -rf core/build core/build-san
	rm -rf $(PY_DIR)/build $(PY_DIR)/*.egg-info .pytest_cache
	rm -f $(PY_LIB_DIR)/libgeozl_kernels* $(PY_LIB_DIR)/geozl_kernels*.dll
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

help:
	@echo "make            build, stage the kernels lib, editable install, smoke load"
	@echo "make build      build libgeozl.a and the kernels lib only"
	@echo "make test       run the C tests under test/ then pytest"
	@echo "make test-c     run the C tests only, builds straight from source"
	@echo "make test-san   run both suites against an ASan and UBSan build"
	@echo "make fuzz       build and run both fuzzers, report in fuzz/out (needs clang)"
	@echo "make fuzz-build build the fuzzers without running them"
	@echo "make clean-fuzz remove fuzz output, corpus and build tree"
	@echo "make test-exhaustive  walk every float32 bit pattern, minutes"
	@echo "make install    cmake --install into PREFIX ($(PREFIX))"
	@echo "make clean      remove all build output, caches and generated files"
	@echo "make submodules fetch or update OpenZL (zstd + lz4)"
	@echo ""
	@echo "vars  BUILD=Debug  PYTHON=python3.12  GEN='Unix Makefiles'  PREFIX=/opt"
	@echo "      FULL=OFF (kernels lib only, what the wheel needs, no OpenZL compile)"
	@echo "      SAN=ON (ASan and UBSan, Linux and macOS, separate build-san tree)"
	@echo "      FUZZ_TIME=600 FUZZ_JOBS=8 (seconds per fuzz target, worker count)"
	@echo "      CMAKE_FLAGS=-DGEOZL_USE_SYSTEM_OPENZL=ON"