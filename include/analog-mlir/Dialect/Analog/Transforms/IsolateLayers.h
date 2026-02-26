#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ISOLATE_LAYERS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_ISOLATE_LAYERS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace analog {

struct IsolateLayersPass
    : public mlir::PassWrapper<IsolateLayersPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsolateLayersPass)

  IsolateLayersPass() = default;
  IsolateLayersPass(const IsolateLayersPass &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createIsolateLayersPass();

} // namespace analog
} // namespace mlir

#endif
