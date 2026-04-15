add_llvm_library(MLIRAnalogRuntime
  lib/TaskGraphRuntime.cpp
)

target_include_directories(MLIRAnalogRuntime PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
)
