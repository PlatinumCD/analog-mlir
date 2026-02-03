#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeWeightsFromConst.h"
#include "analog-mlir/Dialect/Analog/Transforms/MaterializeTilesFromWeights.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace analog {

void registerAnalogPasses() {
  PassRegistration<MaterializeWeightsFromConstPass>();
  PassRegistration<MaterializeTilesFromWeightsPass>();
}

} // namespace analog
} // namespace mlir
