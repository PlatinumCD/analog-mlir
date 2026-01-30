#include "analog-mlir/Transforms/NN/Passes.h"
#include "analog-mlir/Transforms/NN/TagNNLayerIndex.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nn {

void registerNNPasses() {
  PassRegistration<TagNNLayerIndexPass>();
}

} // namespace nn
} // namespace mlir
