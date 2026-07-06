PYTHON ?= python
PREFIX ?= /usr/local
BUILD  ?= Release
GEN    ?= Ninja
FULL   ?= ON

CORE       := core
BUILD_DIR  := core/build
OPENZL     := extern/openzl
PY_DIR     := bindings/python
PY_LIB_DIR := $(PY_DIR)/geozl/_lib

# Kernels lib name per platform, the file the wheel dlopens.
ifeq ($(shell uname -s),Darwin)
  KERNELS := libgeozl_kernels.dylib
else ifeq ($(OS),Windows_NT)
  KERNELS := geozl_kernels.dll
else
  KERNELS := libgeozl_kernels.so
endif

CMAKE_FLAGS ?=
CMAKE_OPTS  := -G $(GEN) -DCMAKE_BUILD_TYPE=$(BUILD) \
               -DGEOZL_BUILD_FULL=$(FULL) -DGEOZL_BUILD_KERNELS_SHARED=ON $(CMAKE_FLAGS)

.PHONY: all build configure lib python test install submodules clean help

# Build the libraries, stage the kernels lib, install the binding editable and
# smoke load it. Run the tests with `make test`.
all: python

# OpenZL carries its own zstd and lz4. A fresh clone has an empty submodule, so
# fetch it on demand. That is what lets a bare `make` work from a clean tree.
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

# Stage the kernels lib beside the binding for cffi. libgeozl.a stays in the
# build tree, a C consumer links it from there.
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
	$(PYTHON) -c "import geozl; print('geozl', geozl.__version__)"

# pytest.
test: python
	@$(PYTHON) -c 'import pytest' 2>/dev/null || { echo "pytest not installed"; exit 1; }
	@$(PYTHON) -m pytest -q $(PY_DIR); \
	  rc=$$?; if [ $$rc -eq 5 ]; then echo "no tests collected"; elif [ $$rc -ne 0 ]; then exit $$rc; fi

install: build
	cmake --install $(BUILD_DIR) --prefix $(PREFIX)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PY_LIB_DIR)/libgeozl_kernels* $(PY_LIB_DIR)/geozl_kernels*.dll

help:
	@echo "make            build, stage the kernels lib, editable install, smoke load"
	@echo "make build      build libgeozl.a and the kernels lib only"
	@echo "make test       run pytest"
	@echo "make install    cmake --install into PREFIX ($(PREFIX))"
	@echo "make clean      remove the build dir and staged libs"
	@echo "make submodules fetch or update OpenZL (zstd + lz4)"
	@echo ""
	@echo "vars  BUILD=Debug  PYTHON=python3.12  GEN='Unix Makefiles'  PREFIX=/opt"
	@echo "      FULL=OFF (kernels lib only, what the wheel needs, no OpenZL compile)"
	@echo "      CMAKE_FLAGS=-DGEOZL_USE_SYSTEM_OPENZL=ON"