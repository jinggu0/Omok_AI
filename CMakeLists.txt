cmake_minimum_required(VERSION 3.24)
project(DojoZero VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DOJOZERO_BUILD_TESTS "Build unit tests" ON)
option(DOJOZERO_BUILD_TOOLS "Build CLI and benchmark tools" ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# Pinned third-party versions (ARCHITECTURE.md §9).
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

add_subdirectory(cpp)

if(DOJOZERO_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()
