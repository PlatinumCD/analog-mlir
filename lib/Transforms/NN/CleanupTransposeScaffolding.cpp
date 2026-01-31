#include "analog-mlir/Transforms/NN/CleanupTransposeScaffolding.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace nn {

// =====--------------------------------=====
//   CleanupTransposeScaffoldingPass - Pass
// =====--------------------------------=====

llvm::StringRef CleanupTransposeScaffoldingPass::getArgument() const {
  return "nn-cleanup-transpose-scaffolding";
}

llvm::StringRef CleanupTransposeScaffoldingPass::getDescription() const {
  return "Remove transpose operations based on layer tagging.";
}

void CleanupTransposeScaffoldingPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CleanupTaggedWeightTranspose>(&getContext());

  if (failed(mlir::applyPatternsGreedily(
          getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createCleanupTransposeScaffoldingPass() {
  return std::make_unique<CleanupTransposeScaffoldingPass>();
}


// =====-------------------------------=====
//  CleanupTaggedWeightTranspose - Rewrite 
// =====-------------------------------=====

LogicalResult CleanupTaggedWeightTranspose::matchAndRewrite(
          mlir::linalg::TransposeOp op,
          PatternRewriter &rewriter) const {

  llvm::errs() << op << "\n";

  Value in = op.getInput();
  if (Operation *def = in.getDefiningOp()) {
    llvm::errs() << "Input comes from:\n\t" << *def << "\n";
    auto tensorTy = llvm::dyn_cast<RankedTensorType>(in.getType());
    if (!tensorTy)
      return success();

    auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(def);
    if (!cst)
      return success();

    if (!llvm::isa<DenseResourceElementsAttr>(cst.getValue()))
      return success();

    if (def->hasAttr("nn.layer_index")) {
        llvm::errs() << "Input Attribute: True" << "\n";

        def->setAttr(
            "nn.transpose",
            IntegerAttr::get(IntegerType::get(getContext(), 64), 1));
    }
  }

  Value out = op.getDpsInitOperand(0)->get();
  if (Operation *def = out.getDefiningOp()) {
    llvm::errs() << "Output comes from:\n\t" << *def << "\n";
//    rewriter.eraseOp();
  }

  llvm::errs() << "\n\n=====================================\n\n"; 
  return success();
}


} // namespace nn
} // namespace mlir
