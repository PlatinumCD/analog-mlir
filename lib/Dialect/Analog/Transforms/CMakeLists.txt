add_mlir_library(MLIRAnalogTransforms
  Passes.cpp
  MaterializeMatrixFromTensor.cpp
  MaterializeVectorFromTensor.cpp
  PartitionMatrix.cpp
  PartitionVector.cpp
  PlaceMatrices.cpp
  PlaceVectors.cpp

  ExecuteArray.cpp
  CombineArrayResults.cpp
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

