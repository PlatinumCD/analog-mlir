#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv1DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv2DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteGroupedConv2DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/RewriteConv3DToMatmul.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/Transforms/IsolateLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionMatrix.h"
#include "analog-mlir/Dialect/Analog/Transforms/PartitionVector.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceMatrices.h"
#include "analog-mlir/Dialect/Analog/Transforms/PlaceVectors.h"

#include "analog-mlir/Dialect/Analog/Transforms/GroupedPasses.h"

#include "analog-mlir/Dialect/Analog/Transforms/ExecuteArray.h"
#include "analog-mlir/Dialect/Analog/Transforms/DispatchLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/DispatchWeights.h"
#include "analog-mlir/Dialect/Analog/Transforms/ReduceResults.h"
#include "analog-mlir/Dialect/Analog/Transforms/ReplaceMatmul.h"

#include <mlir/Pass/PassRegistry.h>


namespace mlir {
namespace analog {

void registerAnalogPasses() {

  // Leaf passes ONLY
  PassRegistration<RewriteConv1DToMatmulPass>();
  PassRegistration<RewriteConv2DToMatmulPass>();
  PassRegistration<RewriteGroupedConv2DToMatmulPass>();
  PassRegistration<RewriteConv3DToMatmulPass>();
  PassRegistration<MaterializeMatrixFromTensorPass>();
  PassRegistration<MaterializeVectorFromTensorPass>();
  PassRegistration<IsolateLayersPass>();
  PassRegistration<PartitionMatrixPass>();
  PassRegistration<PartitionVectorPass>();
  PassRegistration<PlaceMatricesPass>();
  PassRegistration<PlaceVectorsPass>();
  PassRegistration<ExecuteArrayPass>();
  PassRegistration<DispatchLayersPass>();
  PassRegistration<DispatchWeightsPass>();
  PassRegistration<ReduceResultsPass>();
  PassRegistration<ReplaceMatmulPass>();
  
  // Pipelines
  registerRewriteConvToMatmulPipeline();
  registerMaterializeAndPlacePipeline();
  registerExecuteAndReplacePipeline();
  registerDispatchRuntimePipeline();
}

} // namespace analog
} // namespace mlir
