#include "analog-mlir/Dialect/Analog/Transforms/ReplaceMatmul.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/Support/Casting.h"
#include <mlir/IR/DialectRegistry.h>

using namespace mlir;

namespace mlir {
namespace analog {

// =====--------------------------------=====
//   ReplaceMatmulPass - Pass
// =====--------------------------------=====


// Finds the nearest earlier tensor materialization that matches the matmul
// result type and can safely replace the op result.
static bufferization::ToTensorOp
findReplacementTensorForMatmul(linalg::MatmulOp op) {
  for (Operation *candidate = op->getPrevNode(); candidate;
       candidate = candidate->getPrevNode()) {
    auto toTensor = dyn_cast<bufferization::ToTensorOp>(candidate);
    if (!toTensor)
      continue;

    if (toTensor.getResult().getType() == op.getResult(0).getType())
      return toTensor;
  }

  return {};
}


// Returns the command-line name used to invoke this pass from tooling
// and pass pipelines.
llvm::StringRef ReplaceMatmulPass::getArgument() const {
  return "analog-replace-matmul";
}


// Describes the pass as replacing eligible matmul results with earlier
// tensor materializations from the analog path.
llvm::StringRef ReplaceMatmulPass::getDescription() const {
  return "Replace matmuls with analog implementation";
}


// Replaces matmul results with compatible tensors that were already
// materialized earlier in the block.
void ReplaceMatmulPass::runOnOperation() {
  auto func = getOperation();
  bool hadError = false;

  func.walk([&](linalg::MatmulOp op) {
    if (hadError)
      return;

    bufferization::ToTensorOp toTensor = findReplacementTensorForMatmul(op);

    if (!toTensor)
      return;

    Value replacement = toTensor.getResult();

    if (replacement.getType() != op.getResult(0).getType()) {
      op.emitError("replacement tensor type does not match matmul result type");
      hadError = true;
      return;
    }

    op.getResult(0).replaceAllUsesWith(replacement);
    op.erase();
  });

  if (hadError) {
    signalPassFailure();
  }
}


// Registers the dialects required by the ops this pass inspects and
// preserves during rewriting.
void ReplaceMatmulPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
}


// Creates the pass instance used by registration and pipeline builders
// throughout the project.
std::unique_ptr<mlir::Pass> createReplaceMatmulPass() {
  return std::make_unique<ReplaceMatmulPass>();
}

} // namespace analog
} // namespace mlir
