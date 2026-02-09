#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceTiles.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceVTiles.h"

#include "analog-mlir/Dialect/Analog/Transforms/GroupedPasses.h"


#include "analog-mlir/Dialect/Analog/Transforms/IntroduceAnalogOps.h"
#include <mlir/Pass/PassRegistry.h>


namespace mlir {
namespace analog {

void registerAnalogPasses() {

  // Leaf passes ONLY
  PassRegistration<MaterializeMatrixFromTensorPass>();
  PassRegistration<MaterializeVectorFromTensorPass>();
  PassRegistration<PartitionMatrixPass>();
  PassRegistration<PartitionVectorPass>();
  PassRegistration<PlaceTilesPass>();
  PassRegistration<PlaceVTilesPass>();
  PassRegistration<IntroduceAnalogOpsPass>();
  
  // Pipelines
  registerMaterializePipeline();
  registerPartitionPipeline();
  registerPlacePipeline();
}

} // namespace analog
} // namespace mlir
