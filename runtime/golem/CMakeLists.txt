cmake_minimum_required(VERSION 3.16)

project(MLIRAnalogRuntime LANGUAGES CXX)

if(NOT DEFINED NUM_CORES)
  message(FATAL_ERROR "NUM_CORES must be provided when configuring MLIRAnalogRuntime")
endif()

find_package(OpenMP REQUIRED)

add_library(MLIRAnalogRuntime STATIC
  lib/TaskGraphRuntime.cpp
)

target_include_directories(MLIRAnalogRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(MLIRAnalogRuntime PUBLIC cxx_std_17)

target_compile_definitions(MLIRAnalogRuntime PRIVATE
  NUM_CORES=${NUM_CORES}
)

target_link_libraries(MLIRAnalogRuntime PUBLIC
  OpenMP::OpenMP_CXX
)
