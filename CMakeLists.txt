cmake_minimum_required(VERSION 3.21)
project(geozl_root LANGUAGES C)

# Thin top level so the repo configures from the root too. The real build is in
# core/, which enables C++ itself when it needs OpenZL. Consumers that vendor
# geozl add_subdirectory(geozl/core) directly and never see this file.
add_subdirectory(core)