#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REWRITE_CONV3D_TO_MATMUL_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_REWRITE_CONV3D_TO_MATMUL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/IR/DialectRegistry.h>

namespace mlir {
namespace analog {

struct RewriteConv3DToMatmulPass
    : public mlir::PassWrapper<RewriteConv3DToMatmulPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteConv3DToMatmulPass)

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<mlir::Pass> createRewriteConv3DToMatmulPass();

} // namespace analog
} // namespace mlir

#endif
