#include "analog-mlir/Dialect/Analog/Transforms/ExtractLayers.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace analog {

// Outlines layer-shaped regions in forward through the registered extractor set.
void ExtractLayersPass::runOnOperation() {

  // Build the extractor registry in the order patterns should inspect forward.
  mlir::analog::LayerExtractors extractors;
  mlir::analog::registerConv1DExtractor(extractors, &getContext());
  mlir::analog::registerConv2DExtractor(extractors, &getContext());
  mlir::analog::registerConv2DGroupedExtractor(extractors, &getContext());
  mlir::analog::registerConv3DExtractor(extractors, &getContext());
  mlir::analog::registerRNNCellExtractor(extractors, &getContext());
  mlir::analog::registerLinearExtractor(extractors, &getContext());

  // Only forward is treated as the source function for layer outlining.
  for (mlir::func::FuncOp func : getOperation().getOps<mlir::func::FuncOp>()) {
    if (func.getName() != "forward")
      continue;

    // Let each extractor rewrite all matches it owns before the next family runs.
    for (const auto &extractor : extractors) {
      extractor->extract(func);
    }
  }
}

// Registers the layer extraction pass with MLIR's global pass registry.
void registerExtractLayersPass() { PassRegistration<ExtractLayersPass>(); }

} // namespace analog
} // namespace mlir
