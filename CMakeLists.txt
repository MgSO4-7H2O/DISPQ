cmake_minimum_required(VERSION 3.16)
project(ann LANGUAGES CXX)

# Keep single-config builds optimized by default while retaining debug symbols
# for profiling.  Multi-config generators select the configuration at build
# time and should not be overridden here.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Build type: Debug, Release, RelWithDebInfo, or MinSizeRel" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               Debug Release RelWithDebInfo MinSizeRel)
endif()

option(ANN_ENABLE_PROFILING "Enable profiling-friendly compiler options" OFF)

option(ANN_ENABLE_NATIVE
       "Enable host-specific optimization and ISA extensions (e.g. AVX-512)" ON)

if(ANN_ENABLE_NATIVE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    )
  endif()
endif()

if(ANN_ENABLE_PROFILING)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    )
    add_link_options(
      -fno-omit-frame-pointer
    )
  endif()
endif()

include(cmake/Options.cmake)

set(ANN_VENDORED_EIGEN_DIR
    "${CMAKE_SOURCE_DIR}/third_party/eigen"
    CACHE PATH "Path to a vendored Eigen tree; empty to use system Eigen")
if(ANN_VENDORED_EIGEN_DIR AND
   EXISTS "${ANN_VENDORED_EIGEN_DIR}/Eigen/Core")
  add_library(ann_eigen INTERFACE)
  target_include_directories(ann_eigen SYSTEM INTERFACE
                             "${ANN_VENDORED_EIGEN_DIR}")
  add_library(Eigen3::Eigen ALIAS ann_eigen)
  message(STATUS "Using vendored Eigen: ${ANN_VENDORED_EIGEN_DIR}")
else()
  find_package(Eigen3 REQUIRED)
  message(STATUS "Using system Eigen: ${Eigen3_DIR}")
endif()
find_package(OpenMP REQUIRED)

add_library(ann
  src/common/dataset.cpp
  src/common/config.cpp
  src/common/serialization.cpp
  src/whitening/whitening.cpp
  src/index/ivf.cpp
  src/index/soa_kernels.cpp
  src/index/merge.cpp
  src/index/postings.cpp
  src/search/hybrid_search.cpp
  src/search/exact_search.cpp
  src/search/rerank.cpp
  src/eval/gt.cpp
  src/eval/metrics.cpp
)

target_include_directories(ann PUBLIC include third_party)
target_link_libraries(ann PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
ann_set_warnings(ann)

add_executable(build_index apps/build_index.cpp)
target_link_libraries(build_index PRIVATE ann)
ann_set_warnings(build_index)

foreach(eval_target run_eval_large_streaming run_eval run_eval_ms)
  add_executable(${eval_target} apps/run_eval_large_streaming.cpp)
  target_link_libraries(${eval_target} PRIVATE ann)
  ann_set_warnings(${eval_target})
endforeach()

add_executable(test_interfaces tests/test_interfaces.cpp)
target_link_libraries(test_interfaces PRIVATE ann)
ann_set_warnings(test_interfaces)

add_executable(test_recall_smoke tests/test_recall_smoke.cpp)
target_link_libraries(test_recall_smoke PRIVATE ann)
ann_set_warnings(test_recall_smoke)

add_executable(test_online_pq tests/test_online_pq.cpp)
target_link_libraries(test_online_pq PRIVATE ann)
ann_set_warnings(test_online_pq)

add_executable(test_merge tests/test_merge.cpp)
target_link_libraries(test_merge PRIVATE ann)
ann_set_warnings(test_merge)

add_executable(test_soa_kernels tests/test_soa_kernels.cpp)
target_include_directories(test_soa_kernels PRIVATE src/index)
target_link_libraries(test_soa_kernels PRIVATE ann)
ann_set_warnings(test_soa_kernels)

enable_testing()
add_test(NAME test_interfaces COMMAND test_interfaces)
add_test(NAME test_recall_smoke COMMAND test_recall_smoke)
add_test(NAME test_online_pq COMMAND test_online_pq)
add_test(NAME test_merge COMMAND test_merge)
add_test(NAME test_soa_kernels COMMAND test_soa_kernels)
