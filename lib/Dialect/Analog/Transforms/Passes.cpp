#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeWeightsFromConst.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeTilesFromWeights.h"
#include "analog-mlir/Dialect/Analog/Transforms/IntroduceAnalogOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace analog {

void registerAnalogPasses() {
  PassRegistration<MaterializeWeightsFromConstPass>();
  PassRegistration<MaterializeTilesFromWeightsPass>();
  PassRegistration<IntroduceAnalogOpsPass>();
}

} // namespace analog
} // namespace mlir
