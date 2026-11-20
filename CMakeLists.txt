cmake_minimum_required(VERSION 3.21)
project(geozl_root LANGUAGES C)

# core/ defaults to Release only as the top level, which it is not from here.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The real build is in core/. Consumers vendor geozl/core directly.
add_subdirectory(core)