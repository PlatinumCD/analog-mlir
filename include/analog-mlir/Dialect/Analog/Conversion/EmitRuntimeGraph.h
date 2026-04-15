#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_EMITRUNTIMEGRAPH_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_EMITRUNTIMEGRAPH_H

#include "analog-mlir/Dialect/Analog/IR/AnalogDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace analog {

struct EmitRuntimeGraphPass
    : public mlir::PassWrapper<EmitRuntimeGraphPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitRuntimeGraphPass)

  mlir::StringRef getArgument() const final {
    return "analog-emit-runtime-graph";
  }

  mlir::StringRef getDescription() const final {
    return "Emit generic runtime graph metadata and task-entry shims";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::analog::AnalogDialect, mlir::func::FuncDialect,
                    mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

void registerEmitRuntimeGraphPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_EMITRUNTIMEGRAPH_H
