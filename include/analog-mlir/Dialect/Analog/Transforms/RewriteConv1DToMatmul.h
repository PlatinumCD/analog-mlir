#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REWRITE_CONV1D_TO_MATMUL_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REWRITE_CONV1D_TO_MATMUL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/IR/DialectRegistry.h>

namespace mlir {
namespace analog {

struct RewriteConv1DToMatmulPass
    : public mlir::PassWrapper<RewriteConv1DToMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteConv1DToMatmulPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createRewriteConv1DToMatmulPass();

} // namespace analog
} // namespace mlir

#endif
