cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(eval-csp-simulator)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(eval-csp-simulator eval.cpp)
target_link_libraries(eval-csp-simulator "${TORCH_LIBRARIES}")
set_property(TARGET eval-csp-simulator PROPERTY CXX_STANDARD 14)
