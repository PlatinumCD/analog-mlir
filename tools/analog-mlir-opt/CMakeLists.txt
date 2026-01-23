add_llvm_executable(analog-mlir-opt analog-mlir-opt.cpp)

install(TARGETS analog-mlir-opt
EXPORT TorchMLIRTargets
RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
COMPONENT analog-mlir-opt)

set(dependency_libraries)

target_link_libraries(analog-mlir-opt PRIVATE
  MLIROptLib
  MLIRTransforms
  ${dependency_libraries}
)
