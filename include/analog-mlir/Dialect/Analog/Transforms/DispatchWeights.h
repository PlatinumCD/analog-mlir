#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_DISPATCH_WEIGHTS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_DISPATCH_WEIGHTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace analog {

struct DispatchWeightsPass
    : public mlir::PassWrapper<DispatchWeightsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DispatchWeightsPass)

  DispatchWeightsPass() = default;
  DispatchWeightsPass(const DispatchWeightsPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createDispatchWeightsPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_DISPATCH_WEIGHTS_H
