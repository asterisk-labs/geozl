cmake_minimum_required(VERSION 3.21)
project(geozl_root LANGUAGES C)

# The real build is in core/. Consumers vendor geozl/core directly.
add_subdirectory(core)