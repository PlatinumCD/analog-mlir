#include "analog-mlir/Dialect/Analog/Transforms/Passes.h"
#include "analog-mlir/Dialect/Analog/Transforms/AssembleTaskGraph.h"
#include "analog-mlir/Dialect/Analog/Transforms/ConvertLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/ExtractLayers.h"
#include "analog-mlir/Dialect/Analog/Transforms/IsolateWeights.h"

namespace mlir {
namespace analog {

// Registers the transform pass bundle exposed by this library entry point.
void registerAnalogPasses() {
  registerAssembleTaskGraphPass();
  registerConvertLayersPass();
  registerExtractLayersPass();
  registerIsolateWeightsPass();
}

} // namespace analog
} // namespace mlir
