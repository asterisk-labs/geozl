cmake_minimum_required(VERSION 3.21)

project(geozl
    DESCRIPTION "Geospatial codec extensions for OpenZL"
    LANGUAGES   C
)

option(GEOZL_BUILD_FULL    "Build libgeozl, the decoders linked against OpenZL" ON)
option(GEOZL_BUILD_KERNELS "Build libgeozl_kernels, pure C for the cffi layer"  ON)
option(GEOZL_USE_SYSTEM_OPENZL "Link an installed OpenZL instead of the submodule" OFF)

set(GEOZL_OPENZL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/openzl"
    CACHE PATH "Path to the OpenZL submodule source")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The codecs and the registration shim are C11. C++ is enabled only for the
# full build, where OpenZL needs a standard to build against, so the pure C
# kernels lib configures without a C++ compiler.
set(CMAKE_C_STANDARD            11)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_C_EXTENSIONS          OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Public symbols carry GEOZL_API. Hiding the rest keeps the vendored OpenZL and
# its zstd out of the dynamic symbol table.
set(CMAKE_C_VISIBILITY_PRESET       hidden)
set(CMAKE_CXX_VISIBILITY_PRESET     hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(GNUInstallDirs)

if(GEOZL_BUILD_FULL)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD          17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    if(GEOZL_USE_SYSTEM_OPENZL)
        find_package(openzl CONFIG REQUIRED)
        set(GEOZL_OPENZL_TARGET OpenZL::openzl)
    else()
        set(OPENZL_BUILD_TESTS          OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_BENCHMARKS     OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_CLI            OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_EXAMPLES       OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_TOOLS          OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_PYTHON_EXT     OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_CUSTOM_PARSERS OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_CPP            OFF CACHE BOOL "" FORCE)
        set(OPENZL_ALLOW_INTROSPECTION  OFF CACHE BOOL "" FORCE)
        set(OPENZL_INSTALL              OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
        add_subdirectory(${GEOZL_OPENZL_DIR}
                         ${CMAKE_CURRENT_BINARY_DIR}/openzl EXCLUDE_FROM_ALL)
        set(GEOZL_OPENZL_TARGET openzl)
    endif()
endif()

# Decode path only. The encode path runs from Python, so the encode bindings
# stay out of the decode library. The encode kernels build into the kernels lib
# below, for cffi.
set(GEOZL_DECODE_SOURCES
    src/register_lossless.c
    src/register_lossy.c
    src/register_decoders.c
    src/delta_w/decode_delta_w_binding.c
    src/delta_w/decode_delta_w_kernel.c
    src/delta_n/decode_delta_n_binding.c
    src/delta_n/decode_delta_n_kernel.c
    src/planar/decode_planar_binding.c
    src/planar/decode_planar_kernel.c
    src/med/decode_med_binding.c
    src/med/decode_med_kernel.c
    src/average/decode_average_binding.c
    src/average/decode_average_kernel.c
    src/quant_linear/decode_quant_linear_binding.c
    src/quant_linear/decode_quant_linear_kernel.c
)

set(GEOZL_KERNEL_SOURCES
    src/kernels.c
    src/delta_w/encode_delta_w_kernel.c
    src/delta_w/decode_delta_w_kernel.c
    src/delta_n/encode_delta_n_kernel.c
    src/delta_n/decode_delta_n_kernel.c
    src/planar/encode_planar_kernel.c
    src/planar/decode_planar_kernel.c
    src/med/encode_med_kernel.c
    src/med/decode_med_kernel.c
    src/average/encode_average_kernel.c
    src/average/decode_average_kernel.c
    src/quant_linear/encode_quant_linear_kernel.c
    src/quant_linear/decode_quant_linear_kernel.c
)

# OpenZL is PUBLIC because a C consumer runs the decode loop itself and needs
# zl_decompress.h. The folders include each other by relative path, so the
# source dir is on the private include path.
if(GEOZL_BUILD_FULL)
    add_library(geozl ${GEOZL_DECODE_SOURCES})
    add_library(geozl::geozl ALIAS geozl)
    target_include_directories(geozl
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(geozl PUBLIC ${GEOZL_OPENZL_TARGET})
    target_compile_definitions(geozl PRIVATE GEOZL_BUILD)
endif()

# What the wheel ships and cffi loads. It links nothing from OpenZL, but it does
# need the OpenZL headers at compile time, because the quant_linear kernel reads
# ql_dtype from graph_quant_linear.h, which includes openzl/zl_data.h. Headers
# only, the resulting .so has no OpenZL, GDAL or zstd dependency.
if(GEOZL_BUILD_KERNELS)
    add_library(geozl_kernels SHARED ${GEOZL_KERNEL_SOURCES})
    add_library(geozl::kernels ALIAS geozl_kernels)
    target_include_directories(geozl_kernels
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${GEOZL_OPENZL_DIR}/include
    )
    target_compile_definitions(geozl_kernels PRIVATE GEOZL_BUILD)
endif()

# Install the static lib, the kernels lib and the public headers. A consumer
# that vendors geozl with add_subdirectory skips this, it is for a system
# install. A consumer of the installed static lib links OpenZL as well.
if(GEOZL_BUILD_FULL)
    install(TARGETS geozl ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(GEOZL_BUILD_KERNELS)
    install(TARGETS geozl_kernels
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/geozl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})