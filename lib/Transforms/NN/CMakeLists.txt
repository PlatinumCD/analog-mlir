add_mlir_library(MLIRAnalogNNTransforms
  Passes.cpp
  TagNNLayerIndex.cpp
  CleanupTransposeScaffolding.cpp

  DEPENDS
  MLIRFuncDialect
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
)

