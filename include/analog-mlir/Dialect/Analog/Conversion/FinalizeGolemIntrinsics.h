#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_FINALIZEGOLEMINTRINSICS_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_FINALIZEGOLEMINTRINSICS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace analog {

struct FinalizeGolemIntrinsicsPass
    : public mlir::PassWrapper<FinalizeGolemIntrinsicsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeGolemIntrinsicsPass)

  mlir::StringRef getArgument() const final {
    return "analog-finalize-golem-intrinsics";
  }

  mlir::StringRef getDescription() const final {
    return "Rewrite lowered Golem shim calls into RISC-V LLVM intrinsics";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

void registerFinalizeGolemIntrinsicsPass();

} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_FINALIZEGOLEMINTRINSICS_H
