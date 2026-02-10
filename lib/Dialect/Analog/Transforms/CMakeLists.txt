add_mlir_library(MLIRAnalogTransforms
  Passes.cpp
  MaterializeMatrixFromTensor.cpp
  MaterializeVectorFromTensor.cpp
  PartitionMatrix.cpp
  PartitionVector.cpp
  PlaceTiles.cpp
  PlaceVTiles.cpp

  ExecuteTiles.cpp
  CombineTileResults.cpp

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

