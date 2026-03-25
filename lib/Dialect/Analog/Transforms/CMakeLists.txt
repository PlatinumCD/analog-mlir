add_mlir_library(MLIRAnalogTransforms
  Passes.cpp
  PrepareConv2DToMatmul.cpp
  RewriteConv1DToMatmul.cpp
  RewriteConv2DToMatmul.cpp
  MaterializeMatrixFromTensor.cpp
  MaterializeVectorFromTensor.cpp
  IsolateLayers.cpp
  PartitionMatrix.cpp
  PartitionVector.cpp
  PlaceMatrices.cpp
  PlaceVectors.cpp

  ExecuteArray.cpp
  DispatchLayers.cpp
  DispatchWeights.cpp
  ReduceResults.cpp
  ReplaceMatmul.cpp

  GroupedPasses.cpp

  DEPENDS
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
)
