cmake_minimum_required(VERSION 3.21)

project(geozl
    DESCRIPTION "Geospatial codec extensions for OpenZL"
    LANGUAGES   C
)

option(GEOZL_BUILD_FULL           "Build the library that links OpenZL, off builds only the kernels" ON)
option(GEOZL_BUILD_KERNELS_SHARED "Build the pure C kernels shared library for Python" ON)
option(GEOZL_USE_SYSTEM_OPENZL    "Link an installed OpenZL instead of the submodule" OFF)
option(GEOZL_SANITIZE             "Build with ASan and UBSan, for tests and fuzzing" OFF)
option(GEOZL_WERROR               "Turn geozl warnings into errors, for CI" OFF)
option(GEOZL_BUILD_FUZZERS        "Build the libFuzzer harnesses, needs clang and FULL" OFF)
set(GEOZL_OPENZL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/openzl"
    CACHE PATH "Path to the OpenZL submodule source")

# Freeze geozl's linkage before OpenZL touches BUILD_SHARED_LIBS. A standalone
# build is shared; a parent that explicitly sets OFF gets a static library.
# OpenZL may restore an initially undefined value as an empty cache entry, so
# empty still means the standalone default. Use a fresh build tree to change it.
if(NOT DEFINED GEOZL_LIB_TYPE)
    if(DEFINED BUILD_SHARED_LIBS
       AND NOT BUILD_SHARED_LIBS STREQUAL ""
       AND NOT BUILD_SHARED_LIBS)
        set(GEOZL_LIB_TYPE STATIC CACHE INTERNAL "geozl library type")
    else()
        set(GEOZL_LIB_TYPE SHARED CACHE INTERNAL "geozl library type")
    endif()
endif()
message(STATUS "geozl linkage: ${GEOZL_LIB_TYPE}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD                11)
set(CMAKE_C_STANDARD_REQUIRED       ON)
set(CMAKE_C_EXTENSIONS              OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON)

# Hide everything but GEOZL_API so vendored OpenZL stays off the symbol table.
set(CMAKE_C_VISIBILITY_PRESET       hidden)
set(CMAKE_CXX_VISIBILITY_PRESET     hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(GEOZL_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# Coverage instrumentation has to reach the library and OpenZL, not just the
# harness, or libFuzzer has no signal to steer on and mutates blind. Set here,
# before add_subdirectory(openzl), because the decoders are only reachable once
# OpenZL has parsed the frame around them. The harness itself links
# -fsanitize=fuzzer below, which is what brings in libFuzzer's main.
if(GEOZL_BUILD_FUZZERS)
    add_compile_options(-fsanitize=fuzzer-no-link)
    add_link_options(-fsanitize=fuzzer-no-link)
endif()

# A frame is written on one machine and read on another, so the decoder has to
# reconstruct the same bits everywhere. Contraction breaks that: a compiler with
# FMA available folds w*w - offset into one instruction that rounds once instead
# of twice, and whether it does depends on the target and the compiler version.
# On arm64 it always does. Fallin and Burtscher hit the same thing between CPU
# and GPU builds of LC and reached for the same switch (arXiv:2407.15037).
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(GEOZL_FP -ffp-contract=off)
endif()

# Applied to the geozl targets only, so vendored OpenZL keeps its own warnings.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(GEOZL_WARN -Wall -Wextra)
    if(GEOZL_WERROR)
        list(APPEND GEOZL_WARN -Werror)
    endif()
endif()

include(GNUInstallDirs)
find_library(GEOZL_LIBM m)   # the wp_static trainer and the quantizers use libm

# Discover codecs by convention: any src/<name>/ that has an
# encode_<name>_binding.c is a codec. CONFIGURE_DEPENDS re-globs on build, so a
# new codec folder is picked up without editing this file.
file(GLOB GEOZL_CODEC_BINDINGS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*/encode_*_binding.c)
set(GEOZL_CODECS "")
foreach(binding IN LISTS GEOZL_CODEC_BINDINGS)
    get_filename_component(dir ${binding} DIRECTORY)
    get_filename_component(name ${dir} NAME)
    if(EXISTS ${dir}/encode_${name}_binding.c)
        list(APPEND GEOZL_CODECS ${name})
    endif()
endforeach()
list(SORT GEOZL_CODECS)

foreach(name IN LISTS GEOZL_CODECS)
    list(APPEND GEOZL_BINDINGS
        src/${name}/encode_${name}_binding.c
        src/${name}/decode_${name}_binding.c)
    list(APPEND GEOZL_KERNELS
        src/${name}/encode_${name}_kernel.c
        src/${name}/decode_${name}_kernel.c)
endforeach()
# The rest of the kernels library, which the glob above cannot find because none
# of it is a codec. It is here so a Python caller reaches it without OpenZL: the
# coefficient blob's serializer, the wp_static trainer, the recipe parsers that
# turn an error string into a bound, the noise fit the sqrt quantizer needs, and
# the report of which vector paths this build carries.
list(APPEND GEOZL_KERNELS
    src/coeffs/coeffs.c
    src/wp_static/train_wp_static.c
    src/lossy/lossy_recipe.c
    src/quant_linear/quant_linear_spec.c
    src/quant_log/quant_log_spec.c
    src/quant_sqrt/quant_sqrt_spec.c
    src/quant_sqrt/quant_sqrt_fit.c
    src/common/simd.c)

if(GEOZL_BUILD_FULL)
    if(GEOZL_USE_SYSTEM_OPENZL)
        find_package(openzl CONFIG REQUIRED)
        set(GEOZL_OPENZL_TARGET OpenZL::openzl)
    else()
        enable_language(CXX)
        set(CMAKE_CXX_STANDARD          17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        foreach(off TESTS BENCHMARKS CLI EXAMPLES TOOLS PYTHON_EXT CUSTOM_PARSERS CPP)
            set(OPENZL_BUILD_${off} OFF CACHE BOOL "" FORCE)
        endforeach()
        set(OPENZL_ALLOW_INTROSPECTION OFF CACHE BOOL "" FORCE)
        set(OPENZL_INSTALL             OFF CACHE BOOL "" FORCE)
        set(OPENZL_BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
        add_subdirectory(${GEOZL_OPENZL_DIR}
                         ${CMAKE_CURRENT_BINARY_DIR}/openzl EXCLUDE_FROM_ALL)
        set(GEOZL_OPENZL_TARGET openzl)
    endif()

    add_library(geozl ${GEOZL_LIB_TYPE}
        ${GEOZL_BINDINGS}
        ${GEOZL_KERNELS}
        src/2d/2d.c
        src/encoder_registry.c
        src/decoder_registry.c)
    add_library(geozl::geozl ALIAS geozl)

    # OpenZL is PUBLIC, a C consumer runs the graph itself. src is private for the
    # relative includes between folders.
    target_include_directories(geozl
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(geozl PUBLIC ${GEOZL_OPENZL_TARGET})
    target_compile_definitions(geozl PRIVATE GEOZL_BUILD)
    target_compile_options(geozl PRIVATE ${GEOZL_WARN} ${GEOZL_FP})
    if(GEOZL_LIBM)
        target_link_libraries(geozl PRIVATE ${GEOZL_LIBM})
    endif()

    install(TARGETS geozl ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/geozl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # libFuzzer harness over the decode path. Needs clang, and GEOZL_SANITIZE=ON
    # so ASan and UBSan see the decoders. The instrumentation itself is set
    # globally above, this only pulls in libFuzzer's main.
    if(GEOZL_BUILD_FUZZERS)
        foreach(fz decode binding)
            add_executable(geozl_${fz}_fuzzer
                ${CMAKE_CURRENT_SOURCE_DIR}/../fuzz/${fz}_fuzzer.c)
            target_link_libraries(geozl_${fz}_fuzzer PRIVATE geozl)
            target_link_options(geozl_${fz}_fuzzer PRIVATE -fsanitize=fuzzer)
        endforeach()

        # A mutator hardly ever produces a frame that still parses, so the
        # decode fuzzer spends its budget in the OpenZL frame reader and rarely
        # reaches a quantizer at all. These drive the codecs directly and assert
        # the bound on the way back out. Kernels only, no OpenZL.
        foreach(fz roundtrip lossy_recipe quant_linear quant_log quant_sqrt pfor coeffs)
            add_executable(geozl_${fz}_fuzzer
                ${CMAKE_CURRENT_SOURCE_DIR}/../fuzz/${fz}_fuzzer.c
                ${GEOZL_KERNELS})
            target_include_directories(geozl_${fz}_fuzzer PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/src)
            # Without GEOZL_FP the kernels here contract where the library does
            # not, and the fuzzer asserts a bound about arithmetic nobody ships.
            target_compile_options(geozl_${fz}_fuzzer PRIVATE ${GEOZL_FP})
            target_link_options(geozl_${fz}_fuzzer PRIVATE -fsanitize=fuzzer)
            if(GEOZL_LIBM)
                target_link_libraries(geozl_${fz}_fuzzer PRIVATE ${GEOZL_LIBM})
            endif()
        endforeach()
    endif()
endif()

if(GEOZL_BUILD_KERNELS_SHARED)
    add_library(geozl_kernels SHARED ${GEOZL_KERNELS})
    set_target_properties(geozl_kernels PROPERTIES C_VISIBILITY_PRESET default)
    target_compile_definitions(geozl_kernels PRIVATE GEOZL_BUILD)
    # include/ too: the kernels take their parameter blocks from geozl/, which is
    # public because the node builders over them are.
    target_include_directories(geozl_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(geozl_kernels PRIVATE ${GEOZL_WARN} ${GEOZL_FP})
    if(GEOZL_LIBM)
        target_link_libraries(geozl_kernels PRIVATE ${GEOZL_LIBM})
    endif()
    install(TARGETS geozl_kernels LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
