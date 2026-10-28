# Prototype build. `make metal` on macOS, `make cuda` on Linux; SLANGC points at
# slangc. Running needs only the GPU driver, so `make hosts-cuda` builds the
# benchmarks alone for machines without slangc, with PTX built elsewhere.

SLANGC ?= slangc
PYTHON ?= python3
SLANG_VERSION := 2026.19
UNAME := $(shell uname -s)
CORE := ../../core
OPENZL := ../../extern/openzl
KERNELS := ../kernels
SLANG_DEPS := $(shell find $(KERNELS) -name '*.slang')
PFOR_PROGRAM := $(KERNELS)/programs/pfor.slang
PIVCO_PROGRAM := $(KERNELS)/programs/pivco.slang
SLANG_LAYOUT := ../../tools/check_slang_layout.py
entries = $(foreach e,$(1),-entry $(e) -stage compute)
PFOR_ENTRIES := $(call entries,unpack_blocks scan_planes scan_rows scan_columns copy_raw)
PIVCO_ENTRIES := $(call entries,decode_symbols unpack_lane_blocks join_lanes scan_planes scan_rows scan_columns copy_raw)

# planar_zigzag_pfor needs only GeoZL's kernels.
PFOR_SRCS := pfor_bench.c bench.c \
	$(CORE)/src/gpu/pfor_plan.c \
	$(CORE)/src/gpu/pfor_plan_parallel.c \
	$(CORE)/src/pfor/encode_pfor_kernel.c \
	$(CORE)/src/pfor/decode_pfor_kernel.c \
	$(CORE)/src/planar_zigzag/decode_planar_zigzag_kernel.c \
	$(CORE)/src/planar_zigzag_pfor/encode_planar_zigzag_pfor_kernel.c \
	$(CORE)/src/planar_zigzag_pfor/decode_planar_zigzag_pfor_kernel.c \
	$(CORE)/src/common/simd.c
CFLAGS := -O3 -std=c11 -Wall -I$(CORE)/src
ifeq ($(UNAME),Linux)
CFLAGS += -D_POSIX_C_SOURCE=200809L
endif

# planar_zigzag_transpose_pivco runs real frames, so it links a static GeoZL with OpenZL
# inside, where the internal symbols the spy decoder and the planner use are
# reachable.
STATIC := $(CORE)/build-gpu
PIVCO_LIBS := $(STATIC)/libgeozl.a $(STATIC)/openzl/libopenzl.a \
	$(STATIC)/openzl/zstd_build/lib/libzstd.a $(STATIC)/openzl/lz4_build/liblz4.a
PIVCO_SRCS := pivco_bench.c $(CORE)/src/gpu/pivco_capture.c \
	$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pivco_decode.c \
	$(CORE)/src/gpu/pivco_plan_parallel.c \
	$(CORE)/src/gpu/pfor_plan.c $(CORE)/src/gpu/frame_batch.c \
	$(CORE)/src/gpu/staging_arena.c \
	$(CORE)/src/gpu/planar_scan.c bench.c
DECODE_SRCS := decode_bench.c bench.c gpu.c $(CORE)/src/gpu/frame_batch.c \
	$(CORE)/src/gpu/staging_arena.c \
	$(CORE)/src/gpu/pfor_decode.c $(CORE)/src/gpu/pfor_capture.c \
	$(CORE)/src/gpu/pfor_plan.c $(CORE)/src/gpu/pfor_plan_parallel.c \
	$(CORE)/src/gpu/pivco_decode.c $(CORE)/src/gpu/pivco_capture.c \
	$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pivco_plan_parallel.c \
	$(CORE)/src/gpu/planar_scan.c
PIVCO_CFLAGS := -O3 -std=gnu11 -Wall -I$(CORE)/include -I$(CORE)/src \
	-I$(STATIC)/openzl/include -I$(OPENZL)/include -I$(OPENZL)/src
# Each binary links one backend; GPU_BACKEND names it for tests and benchmarks.
METAL_RUNTIME_SRCS := $(CORE)/src/gpu/runtime.c $(CORE)/src/gpu/runtime_metal.m
CUDA_RUNTIME_SRCS := $(CORE)/src/gpu/runtime.c $(CORE)/src/gpu/runtime_cuda.c
METAL_RUNTIME := $(METAL_RUNTIME_SRCS) -DGEOZL_GPU_METAL \
	-DGPU_BACKEND=GEOZL_GPU_BACKEND_METAL
CUDA_RUNTIME := $(CUDA_RUNTIME_SRCS) -DGEOZL_GPU_CUDA \
	-DGPU_BACKEND=GEOZL_GPU_BACKEND_CUDA
GPU_COMMON := gpu.c gpu.h $(CORE)/src/gpu/runtime.h \
	$(CORE)/src/gpu/runtime_internal.h
CHECKS := build/test_bench build/test_pfor_capture build/test_pivco_capture \
	build/test_pivco_plan build/test_planar_scan build/test_runtime_validation
ifeq ($(UNAME),Darwin)
CHECKS += build/test_runtime_metal
endif

.PHONY: metal cuda hosts-cuda check check-metal check-cuda check-slang check-pfor-metal \
	check-gpu-decode \
	check-pivco-metal check-pfor-cuda check-pivco-cuda geozl-static
metal: check-slang build/pfor_metal build/pzp1.metal build/pzp2.metal build/pivco_metal build/pzv1.metal build/pzv2.metal build/decode_metal
cuda: check-slang build/pfor_cuda build/pzp1.ptx build/pzp2.ptx build/pivco_cuda build/pzv1.ptx build/pzv2.ptx build/decode_cuda
hosts-cuda: build/pfor_cuda build/pivco_cuda build/decode_cuda
check-slang:
	@test "$$($(SLANGC) -version 2>&1 | head -n 1)" = "$(SLANG_VERSION)" || \
		{ echo "Slang $(SLANG_VERSION) is required"; exit 1; }
check: $(CHECKS)
	./build/test_bench
	./build/test_pfor_capture
	./build/test_pivco_capture
	./build/test_pivco_plan
	./build/test_planar_scan
	./build/test_runtime_validation
ifeq ($(UNAME),Darwin)
	./build/test_runtime_metal
endif
check-pfor-metal: build/test_pfor_decode_metal build/test_pfor_list_metal build/pzp1.metal build/pzp2.metal
	./build/test_pfor_decode_metal build/pzp1.metal build/pzp2.metal
	./build/test_pfor_list_metal build/pzp1.metal
check-pivco-metal: build/test_pivco_decode_metal build/test_pivco_e2e_metal build/pzv1.metal build/pzv2.metal
	./build/test_pivco_decode_metal build/pzv1.metal
	./build/test_pivco_e2e_metal build/pzv1.metal build/pzv2.metal
check-pfor-cuda: build/test_pfor_decode_cuda build/test_pfor_list_cuda build/pzp1.ptx build/pzp2.ptx
	./build/test_pfor_decode_cuda build/pzp1.ptx build/pzp2.ptx
	./build/test_pfor_list_cuda build/pzp1.ptx
check-pivco-cuda: build/test_pivco_decode_cuda build/test_pivco_e2e_cuda build/pzv1.ptx build/pzv2.ptx
	./build/test_pivco_decode_cuda build/pzv1.ptx
	./build/test_pivco_e2e_cuda build/pzv1.ptx build/pzv2.ptx
check-metal: check-slang check-pfor-metal check-pivco-metal check-gpu-decode
check-cuda: check-slang check-pfor-cuda check-pivco-cuda check-gpu-decode
check-gpu-decode: build/test_gpu_decode
	./build/test_gpu_decode

build:
	mkdir -p build

build/pzp%.metal: $(PFOR_PROGRAM) $(SLANG_DEPS) $(SLANG_LAYOUT) | build
	@test "$$($(SLANGC) -version 2>&1 | head -n 1)" = "$(SLANG_VERSION)" || \
		{ echo "Slang $(SLANG_VERSION) is required"; exit 1; }
	$(SLANGC) $< $(PFOR_ENTRIES) -target metal -D ELT_BYTES=$* \
		-reflection-json $@.json -o $@
	$(PYTHON) $(SLANG_LAYOUT) $@.json pfor
	@touch $@

build/pzp%.ptx: $(PFOR_PROGRAM) $(SLANG_DEPS) $(SLANG_LAYOUT) | build
	@test "$$($(SLANGC) -version 2>&1 | head -n 1)" = "$(SLANG_VERSION)" || \
		{ echo "Slang $(SLANG_VERSION) is required"; exit 1; }
	$(SLANGC) $< $(PFOR_ENTRIES) -target ptx -D ELT_BYTES=$* \
		-reflection-json $@.json -o $@
	$(PYTHON) $(SLANG_LAYOUT) $@.json pfor
	@touch $@

build/pzv%.metal: $(PIVCO_PROGRAM) $(SLANG_DEPS) $(SLANG_LAYOUT) | build
	@test "$$($(SLANGC) -version 2>&1 | head -n 1)" = "$(SLANG_VERSION)" || \
		{ echo "Slang $(SLANG_VERSION) is required"; exit 1; }
	$(SLANGC) $< $(PIVCO_ENTRIES) -target metal -D ELT_BYTES=$* \
		-reflection-json $@.json -o $@
	$(PYTHON) $(SLANG_LAYOUT) $@.json pivco
	@touch $@

build/pzv%.ptx: $(PIVCO_PROGRAM) $(SLANG_DEPS) $(SLANG_LAYOUT) | build
	@test "$$($(SLANGC) -version 2>&1 | head -n 1)" = "$(SLANG_VERSION)" || \
		{ echo "Slang $(SLANG_VERSION) is required"; exit 1; }
	$(SLANGC) $< $(PIVCO_ENTRIES) -target ptx -D ELT_BYTES=$* \
		-reflection-json $@.json -o $@
	$(PYTHON) $(SLANG_LAYOUT) $@.json pivco
	@touch $@

build/pfor_metal: $(PFOR_SRCS) $(GPU_COMMON) \
		$(METAL_RUNTIME_SRCS) bench.h | build
	clang $(CFLAGS) -fobjc-arc $(METAL_RUNTIME) gpu.c \
		$(PFOR_SRCS) -framework Metal -framework Foundation -o $@

build/pfor_cuda: $(PFOR_SRCS) $(GPU_COMMON) \
		$(CUDA_RUNTIME_SRCS) bench.h | build
	cc $(CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) gpu.c \
		$(PFOR_SRCS) -ldl -lpthread -lm -o $@

build/test_bench: test_bench.c bench.c bench.h \
		$(CORE)/src/gpu/pfor_plan.c $(CORE)/src/gpu/pfor_plan_parallel.c \
		$(CORE)/src/gpu/pfor_plan.h | build
	cc $(CFLAGS) test_bench.c bench.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/pfor_plan_parallel.c \
		-lpthread -lm -o $@

build/test_pivco_capture: ../../test/gpu/test_pivco_capture.c \
		../../test/gpu/capture_frame_builder.c \
		../../test/gpu/capture_frame_builder.h \
		$(CORE)/src/gpu/pivco_capture.c $(CORE)/src/gpu/pivco_capture.h \
		geozl-static | build
	cc $(PIVCO_CFLAGS) ../../test/gpu/test_pivco_capture.c \
		../../test/gpu/capture_frame_builder.c \
		$(CORE)/src/gpu/pivco_capture.c $(PIVCO_LIBS) -lm -o $@

build/test_pivco_plan: ../../test/gpu/test_pivco_plan.c \
		$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pivco_plan.h \
		$(CORE)/src/gpu/pivco_plan_parallel.c \
		$(CORE)/src/gpu/pfor_plan.c geozl-static | build
	cc $(PIVCO_CFLAGS) ../../test/gpu/test_pivco_plan.c \
		$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pivco_plan_parallel.c \
		$(CORE)/src/gpu/pfor_plan.c $(PIVCO_LIBS) -lpthread -lm -o $@

build/test_pfor_capture: ../../test/gpu/test_pfor_capture.c \
		../../test/gpu/capture_frame_builder.c \
		../../test/gpu/capture_frame_builder.h \
		$(CORE)/src/gpu/pfor_capture.c $(CORE)/src/gpu/pfor_capture.h \
		geozl-static | build
	cc $(PIVCO_CFLAGS) ../../test/gpu/test_pfor_capture.c \
		../../test/gpu/capture_frame_builder.c \
		$(CORE)/src/gpu/pfor_capture.c $(PIVCO_LIBS) -lm -o $@

build/test_planar_scan: ../../test/gpu/test_planar_scan.c \
		$(CORE)/src/gpu/planar_scan.c $(CORE)/src/gpu/planar_scan.h | build
	cc $(CFLAGS) ../../test/gpu/test_planar_scan.c \
		$(CORE)/src/gpu/planar_scan.c -o $@

build/test_runtime_validation: ../../test/gpu/test_runtime_validation.c \
		$(CORE)/src/gpu/runtime.h $(CORE)/src/gpu/runtime_internal.h | build
	cc $(CFLAGS) ../../test/gpu/test_runtime_validation.c -o $@

build/test_runtime_metal: ../../test/gpu/test_runtime_metal.m \
		$(METAL_RUNTIME_SRCS) $(CORE)/src/gpu/runtime.h \
		$(CORE)/src/gpu/runtime_internal.h $(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/staging_arena.h | build
	clang $(CFLAGS) -fobjc-arc $(METAL_RUNTIME) \
		$(CORE)/src/gpu/staging_arena.c ../../test/gpu/test_runtime_metal.m \
		-framework Metal -framework Foundation -o $@

build/test_pfor_decode_metal: ../../test/gpu/test_pfor_decode.c \
		$(CORE)/src/gpu/pfor_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pfor_capture.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/pfor_plan_parallel.c \
		$(CORE)/src/gpu/planar_scan.c $(METAL_RUNTIME_SRCS) \
		geozl-static | build
	clang $(PIVCO_CFLAGS) -fobjc-arc $(METAL_RUNTIME) \
		$(CORE)/src/gpu/pfor_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pfor_capture.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/pfor_plan_parallel.c \
		$(CORE)/src/gpu/planar_scan.c ../../test/gpu/test_pfor_decode.c \
		$(PIVCO_LIBS) -framework Metal -framework Foundation -o $@

build/test_pfor_list_metal: ../../test/gpu/test_pfor_list.c \
		$(CORE)/src/gpu/pfor_plan.c $(METAL_RUNTIME_SRCS) | build
	clang $(PIVCO_CFLAGS) -fobjc-arc $(METAL_RUNTIME) \
		$(CORE)/src/gpu/pfor_plan.c ../../test/gpu/test_pfor_list.c \
		-framework Metal -framework Foundation -o $@

build/test_pivco_decode_metal: ../../test/gpu/test_pivco_decode.c \
		$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pfor_plan.c \
		$(METAL_RUNTIME_SRCS) geozl-static | build
	clang $(PIVCO_CFLAGS) -fobjc-arc $(METAL_RUNTIME) \
		$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pfor_plan.c \
		../../test/gpu/test_pivco_decode.c $(PIVCO_LIBS) \
		-framework Metal -framework Foundation -o $@

build/test_pivco_e2e_metal: ../../test/gpu/test_pivco_e2e.c \
		$(CORE)/src/gpu/pivco_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pivco_capture.c $(CORE)/src/gpu/pivco_plan.c \
		$(CORE)/src/gpu/pivco_plan_parallel.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/planar_scan.c $(METAL_RUNTIME_SRCS) \
		geozl-static | build
	clang $(PIVCO_CFLAGS) -fobjc-arc $(METAL_RUNTIME) \
		$(CORE)/src/gpu/pivco_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pivco_capture.c $(CORE)/src/gpu/pivco_plan.c \
		$(CORE)/src/gpu/pivco_plan_parallel.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/planar_scan.c ../../test/gpu/test_pivco_e2e.c \
		$(PIVCO_LIBS) -lpthread -lm -framework Metal -framework Foundation -o $@

build/test_pfor_decode_cuda: ../../test/gpu/test_pfor_decode.c \
		$(CORE)/src/gpu/pfor_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pfor_capture.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/pfor_plan_parallel.c \
		$(CORE)/src/gpu/planar_scan.c $(CUDA_RUNTIME_SRCS) \
		geozl-static | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) \
		$(CORE)/src/gpu/pfor_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pfor_capture.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/pfor_plan_parallel.c \
		$(CORE)/src/gpu/planar_scan.c ../../test/gpu/test_pfor_decode.c \
		$(PIVCO_LIBS) -ldl -lpthread -lm -o $@

build/test_pfor_list_cuda: ../../test/gpu/test_pfor_list.c \
		$(CORE)/src/gpu/pfor_plan.c $(CUDA_RUNTIME_SRCS) | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) \
		$(CORE)/src/gpu/pfor_plan.c ../../test/gpu/test_pfor_list.c \
		-ldl -lpthread -lm -o $@

build/test_pivco_decode_cuda: ../../test/gpu/test_pivco_decode.c \
		$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pfor_plan.c \
		$(CUDA_RUNTIME_SRCS) geozl-static | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) \
		$(CORE)/src/gpu/pivco_plan.c $(CORE)/src/gpu/pfor_plan.c \
		../../test/gpu/test_pivco_decode.c $(PIVCO_LIBS) \
		-ldl -lpthread -lm -o $@

build/test_pivco_e2e_cuda: ../../test/gpu/test_pivco_e2e.c \
		$(CORE)/src/gpu/pivco_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pivco_capture.c $(CORE)/src/gpu/pivco_plan.c \
		$(CORE)/src/gpu/pivco_plan_parallel.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/planar_scan.c $(CUDA_RUNTIME_SRCS) \
		geozl-static | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) \
		$(CORE)/src/gpu/pivco_decode.c $(CORE)/src/gpu/frame_batch.c \
		$(CORE)/src/gpu/staging_arena.c \
		$(CORE)/src/gpu/pivco_capture.c $(CORE)/src/gpu/pivco_plan.c \
		$(CORE)/src/gpu/pivco_plan_parallel.c $(CORE)/src/gpu/pfor_plan.c \
		$(CORE)/src/gpu/planar_scan.c ../../test/gpu/test_pivco_e2e.c \
		$(PIVCO_LIBS) -ldl -lpthread -lm -o $@

# cmake decides whether the library is out of date.
geozl-static:
	@test -f $(STATIC)/CMakeCache.txt || cmake -S $(CORE) -B $(STATIC) -DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF -DGEOZL_BUILD_KERNELS_SHARED=OFF -DGEOZL_BUILD_GPU=ON
	cmake --build $(STATIC) --target geozl

build/pivco_metal: $(PIVCO_SRCS) $(GPU_COMMON) \
		$(METAL_RUNTIME_SRCS) bench.h \
		$(CORE)/src/gpu/pivco_capture.h $(CORE)/src/gpu/pivco_plan.h \
		$(CORE)/src/gpu/pivco_decode.h $(CORE)/src/gpu/planar_scan.h \
		geozl-static | build
	clang $(PIVCO_CFLAGS) -fobjc-arc $(METAL_RUNTIME) gpu.c \
		$(PIVCO_SRCS) $(PIVCO_LIBS) -framework Metal -framework Foundation -o $@

build/pivco_cuda: $(PIVCO_SRCS) $(GPU_COMMON) \
		$(CUDA_RUNTIME_SRCS) bench.h \
		$(CORE)/src/gpu/pivco_capture.h $(CORE)/src/gpu/pivco_plan.h \
		$(CORE)/src/gpu/pivco_decode.h $(CORE)/src/gpu/planar_scan.h \
		geozl-static | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) gpu.c \
		$(PIVCO_SRCS) $(PIVCO_LIBS) -ldl -lpthread -lm -o $@

build/decode_metal: $(DECODE_SRCS) $(GPU_COMMON) $(METAL_RUNTIME_SRCS) bench.h \
		$(wildcard $(CORE)/src/gpu/*.h) geozl-static | build
	clang $(PIVCO_CFLAGS) -fobjc-arc $(METAL_RUNTIME) $(DECODE_SRCS) \
		$(PIVCO_LIBS) -lpthread -lm -framework Metal -framework Foundation -o $@

build/decode_cuda: $(DECODE_SRCS) $(GPU_COMMON) $(CUDA_RUNTIME_SRCS) bench.h \
		$(wildcard $(CORE)/src/gpu/*.h) geozl-static | build
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $(CUDA_RUNTIME) $(DECODE_SRCS) \
		$(PIVCO_LIBS) -ldl -lpthread -lm -o $@

# The public decoder, with its kernels inside the static library.
build/test_gpu_decode: ../../test/gpu/test_gpu_decode.c geozl-static | build
ifeq ($(UNAME),Darwin)
	clang $(PIVCO_CFLAGS) $< $(PIVCO_LIBS) -lpthread -lm \
		-framework Metal -framework Foundation -o $@
else
	cc $(PIVCO_CFLAGS) -D_GNU_SOURCE $< $(PIVCO_LIBS) -ldl -lpthread -lm -o $@
endif
