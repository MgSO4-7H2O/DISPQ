cmake_minimum_required(VERSION 3.16)
project(ann LANGUAGES CXX)

include(cmake/Options.cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_library(ann
  src/common/dataset.cpp
  src/common/config.cpp
  src/common/serialization.cpp
  src/whitening/whitening.cpp
  src/index/ivf.cpp
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

add_executable(run_eval apps/run_eval.cpp)
target_link_libraries(run_eval PRIVATE ann)
ann_set_warnings(run_eval)

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

enable_testing()
add_test(NAME test_interfaces COMMAND test_interfaces)
add_test(NAME test_recall_smoke COMMAND test_recall_smoke)
add_test(NAME test_online_pq COMMAND test_online_pq)
add_test(NAME test_merge COMMAND test_merge)
