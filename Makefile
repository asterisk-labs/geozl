# geozl build. Common targets, see `make help` for the rest.
#
#   make            build and install the Python binding, then smoke load it
#   make test       run the test suite
#   make test-san   run the suite under ASan and UBSan
#   make fuzz       build and run the decode fuzzer
#   make clean      remove all build output and generated files
#
# Vendors OpenZL as a submodule, fetched on first build.

PYTHON ?= python
PREFIX ?= /usr/local
BUILD  ?= Release
GEN    ?= Ninja
FULL   ?= ON
SAN    ?= OFF

CORE       := core
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
else ifeq ($(OS),Windows_NT)
  KERNELS := geozl_kernels.dll
else
  KERNELS := libgeozl_kernels.so
endif

CMAKE_FLAGS ?=
CMAKE_OPTS  := -G $(GEN) -DCMAKE_BUILD_TYPE=$(BUILD) \
               -DGEOZL_BUILD_FULL=$(FULL) -DGEOZL_BUILD_KERNELS_SHARED=ON \
               -DGEOZL_SANITIZE=$(SAN) $(CMAKE_FLAGS)

.PHONY: all build configure lib python test test-san fuzz install submodules clean help

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

python: lib
	@$(PYTHON) -c 'import numpy, cffi, openzl' 2>/dev/null \
	  || { echo "missing runtime deps, install: numpy cffi openzl"; exit 1; }
	$(PYTHON) -m pip install -e $(PY_DIR) -q
	$(SAN_ENV) $(PYTHON) -c "import geozl; print('geozl', geozl.__version__)"

test: python
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

fuzz: $(OPENZL)/CMakeLists.txt python
	cmake -S $(CORE) -B core/build-fuzz -G $(GEN) -DCMAKE_BUILD_TYPE=$(BUILD) \
	      -DGEOZL_BUILD_FULL=ON -DGEOZL_SANITIZE=ON -DGEOZL_BUILD_FUZZERS=ON \
	      -DCMAKE_C_COMPILER=$(CLANG) -DCMAKE_CXX_COMPILER=$(CLANG)++
	cmake --build core/build-fuzz --target geozl_decode_fuzzer
	@$(PYTHON) fuzz/gen_corpus.py fuzz/corpus
	@echo "run: core/build-fuzz/geozl_decode_fuzzer fuzz/corpus"

install: build
	cmake --install $(BUILD_DIR) --prefix $(PREFIX)

clean:
	rm -rf core/build core/build-san core/build-fuzz fuzz/corpus
	rm -rf $(PY_DIR)/build $(PY_DIR)/*.egg-info .pytest_cache
	rm -f $(PY_LIB_DIR)/libgeozl_kernels* $(PY_LIB_DIR)/geozl_kernels*.dll
	rm -f crash-* leak-* timeout-* oom-*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

help:
	@echo "make            build, stage the kernels lib, editable install, smoke load"
	@echo "make build      build libgeozl.a and the kernels lib only"
	@echo "make test       run pytest"
	@echo "make test-san   run pytest against an ASan and UBSan build"
	@echo "make fuzz       build the libFuzzer decode fuzzer (needs full LLVM clang)"
	@echo "make install    cmake --install into PREFIX ($(PREFIX))"
	@echo "make clean      remove all build output, caches and generated files"
	@echo "make submodules fetch or update OpenZL (zstd + lz4)"
	@echo ""
	@echo "vars  BUILD=Debug  PYTHON=python3.12  GEN='Unix Makefiles'  PREFIX=/opt"
	@echo "      FULL=OFF (kernels lib only, what the wheel needs, no OpenZL compile)"
	@echo "      SAN=ON (ASan and UBSan, Linux and macOS, separate build-san tree)"
	@echo "      CMAKE_FLAGS=-DGEOZL_USE_SYSTEM_OPENZL=ON"