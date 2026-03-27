#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_IDENTIFY_RECURRENT_PATTERNS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_IDENTIFY_RECURRENT_PATTERNS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/IR/DialectRegistry.h>

namespace mlir {
namespace analog {

struct IdentifyRecurrentPatternsPass
    : public mlir::PassWrapper<IdentifyRecurrentPatternsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      IdentifyRecurrentPatternsPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createIdentifyRecurrentPatternsPass();

} // namespace analog
} // namespace mlir

#endif
