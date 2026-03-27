#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PREPARE_RNN_FOR_ANALOG_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_PREPARE_RNN_FOR_ANALOG_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/IR/DialectRegistry.h>

namespace mlir {
namespace analog {

struct PrepareRNNForAnalogPass
    : public mlir::PassWrapper<PrepareRNNForAnalogPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareRNNForAnalogPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createPrepareRNNForAnalogPass();

} // namespace analog
} // namespace mlir

#endif
