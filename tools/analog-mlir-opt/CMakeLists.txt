add_llvm_executable(analog-mlir-opt analog-mlir-opt.cpp)

install(TARGETS analog-mlir-opt
        EXPORT TorchMLIRTargets
        RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
        COMPONENT analog-mlir-opt)

# Pull in *all* upstream dialect + conversion + extension libraries (the mlir-opt way).
get_property(MLIR_DIALECT_LIBS    GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_EXTENSION_LIBS  GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(analog-mlir-opt PRIVATE
  MLIRAnalogNNTransforms
  ${MLIR_DIALECT_LIBS}
  ${MLIR_CONVERSION_LIBS}
  ${MLIR_EXTENSION_LIBS}

  # Common opt driver + infrastructure libs (again, mirrors mlir-opt style)
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRDialect
)
