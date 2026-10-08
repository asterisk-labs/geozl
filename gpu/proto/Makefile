# Prototype build. `make metal` on macOS, `make cuda` on Linux; SLANGC points at
# slangc. Running needs only the GPU driver, so `make hosts-cuda` builds the
# benchmarks alone for machines without slangc, with PTX built elsewhere.

SLANGC ?= slangc
CORE := ../../core
OPENZL := ../../extern/openzl
KERNELS := ../kernels
SLANG_DEPS := $(wildcard $(KERNELS)/*.slang)
entries = $(foreach e,$(1),-entry $(e) -stage compute)
PFOR_ENTRIES := $(call entries,unpack_blocks scan_planes scan_rows scan_columns)
PIVCO_ENTRIES := $(call entries,build_ranks decode_symbols unpack_lane_blocks join_lanes scan_planes scan_rows scan_columns)

# planar_zigzag_pfor needs only GeoZL's kernels.
PFOR_SRCS := pfor_bench.c bench.c \
	$(CORE)/src/pfor/encode_pfor_kernel.c \
	$(CORE)/src/pfor/decode_pfor_kernel.c \
	$(CORE)/src/planar_zigzag/decode_planar_zigzag_kernel.c \
	$(CORE)/src/planar_zigzag_pfor/encode_planar_zigzag_pfor_kernel.c \
	$(CORE)/src/planar_zigzag_pfor/decode_planar_zigzag_pfor_kernel.c \
	$(CORE)/src/common/simd.c
CFLAGS := -O3 -std=c11 -Wall -I$(CORE)/src

# planar_zigzag_pivco runs real frames, so it links a static GeoZL with OpenZL
# inside, where the internal symbols the spy decoder and the planner use are
# reachable.
STATIC := $(CORE)/build-gpu
PIVCO_LIBS := $(STATIC)/libgeozl.a $(STATIC)/openzl/libopenzl.a \
	$(STATIC)/openzl/zstd_build/lib/libzstd.a $(STATIC)/openzl/lz4_build/liblz4.a
PIVCO_SRCS := pivco_bench.c pivco_plan.c bench.c
PIVCO_CFLAGS := -O3 -std=gnu11 -Wall -I$(CORE)/include -I$(CORE)/src \
	-I$(STATIC)/openzl/include -I$(OPENZL)/include -I$(OPENZL)/src

.PHONY: metal cuda hosts-cuda check geozl-static
metal: build/pfor_metal build/pzp1.metal build/pzp2.metal build/pivco_metal build/pzv1.metal build/pzv2.metal
cuda: build/pfor_cuda build/pzp1.ptx build/pzp2.ptx build/pivco_cuda build/pzv1.ptx build/pzv2.ptx
hosts-cuda: build/pfor_cuda build/pivco_cuda
check: build/test_bench
	./build/test_bench

build:
	mkdir -p build

build/pzp%.metal: $(KERNELS)/planar_zigzag_pfor.slang $(SLANG_DEPS) | build
	$(SLANGC) $< $(PFOR_ENTRIES) -target metal -D ELT_BYTES=$* -o $@

build/pzp%.ptx: $(KERNELS)/planar_zigzag_pfor.slang $(SLANG_DEPS) | build
	$(SLANGC) $< $(PFOR_ENTRIES) -target ptx -D ELT_BYTES=$* -o $@

build/pzv%.metal: $(KERNELS)/planar_zigzag_pivco.slang $(SLANG_DEPS) | build
	$(SLANGC) $< $(PIVCO_ENTRIES) -target metal -D ELT_BYTES=$* -o $@

build/pzv%.ptx: $(KERNELS)/planar_zigzag_pivco.slang $(SLANG_DEPS) | build
	$(SLANGC) $< $(PIVCO_ENTRIES) -target ptx -D ELT_BYTES=$* -o $@

build/pfor_metal: $(PFOR_SRCS) gpu_metal.m gpu.h bench.h | build
	clang $(CFLAGS) -fobjc-arc gpu_metal.m $(PFOR_SRCS) -framework Metal -framework Foundation -o $@

build/pfor_cuda: $(PFOR_SRCS) gpu_cuda.c gpu.h bench.h | build
	cc $(CFLAGS) -D_GNU_SOURCE gpu_cuda.c $(PFOR_SRCS) -ldl -lpthread -lm -o $@

build/test_bench: test_bench.c bench.c bench.h | build
	cc $(CFLAGS) test_bench.c bench.c -lpthread -lm -o $@

# cmake decides whether the library is out of date.
geozl-static:
	@test -f $(STATIC)/CMakeCache.txt || cmake -S $(CORE) -B $(STATIC) -DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF -DGEOZL_BUILD_KERNELS_SHARED=OFF
	cmake --build $(STATIC) --target geozl

build/pivco_metal: $(PIVCO_SRCS) gpu_metal.m gpu.h bench.h pivco_plan.h geozl-static | build
	clang $(PIVCO_CFLAGS) -fobjc-arc gpu_metal.m $(PIVCO_SRCS) $(PIVCO_LIBS) -framework Metal -framework Foundation -o $@

build/pivco_cuda: $(PIVCO_SRCS) gpu_cuda.c gpu.h bench.h pivco_plan.h geozl-static | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE gpu_cuda.c $(PIVCO_SRCS) $(PIVCO_LIBS) -ldl -lpthread -lm -o $@
