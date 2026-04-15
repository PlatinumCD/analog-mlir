#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ISOLATEWEIGHTS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ISOLATEWEIGHTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace analog {

// Splits analog layer weight initialization into private helper functions and
// orders their calls before dependent layers in forward.
struct IsolateWeightsPass
    : public mlir::PassWrapper<IsolateWeightsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsolateWeightsPass)

  // Provides the command-line name used to enable this transform.
  mlir::StringRef getArgument() const final {
    return "analog-isolate-weights";
  }

  // Describes the pass in MLIR pass listings.
  mlir::StringRef getDescription() const final {
    return "Isolate weights into dedicated helper structure";
  }

  // Rewrites the module-level analog layer graph to isolate weight setup.
  void runOnOperation() override;
};

// Makes the isolate-weights pass available to the analog transform pipeline.
void registerIsolateWeightsPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ISOLATEWEIGHTS_H
