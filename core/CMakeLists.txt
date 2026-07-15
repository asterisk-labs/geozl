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
option(GEOZL_BUILD_FUZZERS        "Build the libFuzzer decode fuzzer, needs clang and FULL" OFF)
set(GEOZL_OPENZL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/openzl"
    CACHE PATH "Path to the OpenZL submodule source")

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

# Applied to the geozl targets only, so vendored OpenZL keeps its own warnings.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(GEOZL_WARN -Wall -Wextra)
    if(GEOZL_WERROR)
        list(APPEND GEOZL_WARN -Werror)
    endif()
endif()

include(GNUInstallDirs)
find_library(GEOZL_LIBM m)   # the wp_static trainer uses log2 and lround

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
list(APPEND GEOZL_KERNELS src/wp_static/train_wp_static.c)

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

    add_library(geozl
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
    target_compile_options(geozl PRIVATE ${GEOZL_WARN})
    if(GEOZL_LIBM)
        target_link_libraries(geozl PRIVATE ${GEOZL_LIBM})
    endif()

    install(TARGETS geozl ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/geozl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # libFuzzer harness over the decode path. Needs clang, and GEOZL_SANITIZE=ON
    # so ASan and UBSan see the decoders.
    if(GEOZL_BUILD_FUZZERS)
        add_executable(geozl_decode_fuzzer
            ${CMAKE_CURRENT_SOURCE_DIR}/../fuzz/decode_fuzzer.c)
        target_link_libraries(geozl_decode_fuzzer PRIVATE geozl)
        target_compile_options(geozl_decode_fuzzer PRIVATE -fsanitize=fuzzer)
        target_link_options(geozl_decode_fuzzer PRIVATE -fsanitize=fuzzer)
    endif()
endif()

if(GEOZL_BUILD_KERNELS_SHARED)
    add_library(geozl_kernels SHARED ${GEOZL_KERNELS})
    set_target_properties(geozl_kernels PROPERTIES C_VISIBILITY_PRESET default)
    target_include_directories(geozl_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(geozl_kernels PRIVATE ${GEOZL_WARN})
    if(GEOZL_LIBM)
        target_link_libraries(geozl_kernels PRIVATE ${GEOZL_LIBM})
    endif()
    install(TARGETS geozl_kernels LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()