cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(csp-cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(csp-cpp eval.cpp)
target_link_libraries(csp-cpp "${TORCH_LIBRARIES}")
set_property(TARGET csp-cpp PROPERTY CXX_STANDARD 14)
