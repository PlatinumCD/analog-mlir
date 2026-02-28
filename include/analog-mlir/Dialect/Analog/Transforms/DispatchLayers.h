#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_DISPATCH_LAYERS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_DISPATCH_LAYERS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace analog {

struct DispatchLayersPass
    : public mlir::PassWrapper<DispatchLayersPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DispatchLayersPass)

  DispatchLayersPass() = default;
  DispatchLayersPass(const DispatchLayersPass &other) : PassWrapper(other) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createDispatchLayersPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_DISPATCH_LAYERS_H
