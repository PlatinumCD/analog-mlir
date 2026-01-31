#ifndef ANALOG_MLIR_TRANSFORMS_NN_CLEANUPTRANSPOSESCAFFOLDING_H
#define ANALOG_MLIR_TRANSFORMS_NN_CLEANUPTRANSPOSESCAFFOLDING_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nn {

struct CleanupTransposeScaffoldingPass
    : public mlir::PassWrapper<CleanupTransposeScaffoldingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CleanupTransposeScaffoldingPass);

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  void runOnOperation() override;

};

struct CleanupTaggedWeightTranspose
    : mlir::OpRewritePattern<mlir::linalg::TransposeOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
          mlir::linalg::TransposeOp op,
          PatternRewriter &rewriter) const override;

};

std::unique_ptr<mlir::Pass> createCleanupTransposeScaffoldingPass();


} // namespace nn
} // namespace mlir

#endif
