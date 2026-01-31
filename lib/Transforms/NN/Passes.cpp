#include "analog-mlir/Transforms/NN/Passes.h"
#include "analog-mlir/Transforms/NN/TagNNLayerIndex.h"
#include "analog-mlir/Transforms/NN/CleanupTransposeScaffolding.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nn {

void registerNNPasses() {
  PassRegistration<TagNNLayerIndexPass>();
  PassRegistration<CleanupTransposeScaffoldingPass>();

}

} // namespace nn
} // namespace mlir
