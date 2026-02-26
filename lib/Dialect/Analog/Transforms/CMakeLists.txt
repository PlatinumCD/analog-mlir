add_mlir_library(MLIRAnalogTransforms
  Passes.cpp
  MaterializeMatrixFromTensor.cpp
  MaterializeVectorFromTensor.cpp
  IsolateLayers.cpp
  PartitionMatrix.cpp
  PartitionVector.cpp
  PlaceMatrices.cpp
  PlaceVectors.cpp

  ExecuteArray.cpp
  ReduceResults.cpp
  ReplaceMatmul.cpp

  GroupedPasses.cpp

  DEPENDS
  MLIRFuncDialect
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
)
