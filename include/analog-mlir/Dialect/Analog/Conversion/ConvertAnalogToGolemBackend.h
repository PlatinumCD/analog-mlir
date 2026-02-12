#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERT_ANALOG_TO_LOWER_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERT_ANALOG_TO_LOWER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace analog {

struct ConvertAnalogToGolemBackendPass
    : public mlir::PassWrapper<ConvertAnalogToGolemBackendPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAnalogToGolemBackendPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createConvertAnalogToGolemBackendPass();

} // namespace analog
} // namespace mlir

#endif
