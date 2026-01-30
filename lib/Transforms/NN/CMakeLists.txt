add_mlir_library(MLIRAnalogNNTransforms
  Passes.cpp
  TagNNLayerIndex.cpp

  DEPENDS
  MLIRFuncDialect
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
)

