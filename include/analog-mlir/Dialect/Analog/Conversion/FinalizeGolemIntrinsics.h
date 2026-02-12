#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_FINALIZE_GOLEM_INTRINSICS_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_FINALIZE_GOLEM_INTRINSICS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>

namespace mlir {
namespace analog {

struct FinalizeGolemIntrinsicsPass
    : public mlir::PassWrapper<FinalizeGolemIntrinsicsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeGolemIntrinsicsPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createFinalizeGolemIntrinsicsPass();

} // namespace analog
} // namespace mlir

#endif
