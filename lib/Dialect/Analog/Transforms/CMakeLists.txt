add_mlir_library(MLIRAnalogTransforms
  Passes.cpp
  MaterializeWeightsFromConst.cpp
  MaterializeTilesFromWeights.cpp
  IntroduceAnalogOps.cpp

  DEPENDS
  MLIRFuncDialect
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
)

