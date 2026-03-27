add_mlir_library(MLIRAnalogTransforms
  Passes.cpp
  IdentifyRecurrentPatterns.cpp
  PrepareRNNForAnalog.cpp
  PrepareRNNCellForAnalog.cpp
  RewriteConv1DToMatmul.cpp
  RewriteConv2DToMatmul.cpp
  RewriteGroupedConv2DToMatmul.cpp
  RewriteConv3DToMatmul.cpp
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
