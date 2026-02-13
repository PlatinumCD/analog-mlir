#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERT_ANALOG_TO_DEBUG_SHIMS_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_CONVERT_ANALOG_TO_DEBUG_SHIMS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace analog {

struct ConvertAnalogToDebugShimsPass
    : public mlir::PassWrapper<ConvertAnalogToDebugShimsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAnalogToDebugShimsPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createConvertAnalogToDebugShimsPass();

} // namespace analog
} // namespace mlir

#endif
