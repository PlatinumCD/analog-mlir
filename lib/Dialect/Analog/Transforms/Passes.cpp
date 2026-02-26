#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceMatrices.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceVectors.h"

#include "analog-mlir/Dialect/Analog/Transforms/GroupedPasses.h"

#include "analog-mlir/Dialect/Analog/Transforms/ExecuteArray.h"
#include "analog-mlir/Dialect/Analog/Transforms/CombineArrayResults.h"
#include "analog-mlir/Dialect/Analog/Transforms/ReplaceMatmul.h"

#include <mlir/Pass/PassRegistry.h>


namespace mlir {
namespace analog {

void registerAnalogPasses() {

  // Leaf passes ONLY
  PassRegistration<MaterializeMatrixFromTensorPass>();
  PassRegistration<MaterializeVectorFromTensorPass>();
  PassRegistration<PartitionLayersPass>();
  PassRegistration<PartitionMatrixPass>();
  PassRegistration<PartitionVectorPass>();
  PassRegistration<PlaceMatricesPass>();
  PassRegistration<PlaceVectorsPass>();
  PassRegistration<ExecuteArrayPass>();
  PassRegistration<CombineArrayResultsPass>();
  PassRegistration<ReplaceMatmulPass>();
  
  // Pipelines
  registerMaterializePipeline();
  registerPartitionPipeline();
  registerPlacePipeline();
}

} // namespace analog
} // namespace mlir
