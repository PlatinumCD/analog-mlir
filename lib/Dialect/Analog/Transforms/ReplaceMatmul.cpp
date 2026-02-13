#include "analog-mlir/Dialect/Analog/Transforms/ReplaceMatmul.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <deque>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>


using namespace mlir;

namespace mlir {
namespace analog {

// =====--------------------------------=====
//   ReplaceMatmulPass - Pass
// =====--------------------------------=====

llvm::StringRef ReplaceMatmulPass::getArgument() const {
  return "analog-replace-matmul";
}

llvm::StringRef ReplaceMatmulPass::getDescription() const {
  return "Replace matmuls with analog implementation";
}

void ReplaceMatmulPass::runOnOperation() {
  auto func = getOperation();

  // Find array partition
  func.walk([&](linalg::MatmulOp op) {
    Operation *prev = op->getPrevNode();
    if (!prev)
      return;

    auto toTensor = dyn_cast<bufferization::ToTensorOp>(prev);
    if (!toTensor)
      return;

    Value replacement = toTensor.getResult();

    if (replacement.getType() != op.getResult(0).getType())
      return;

    op.getResult(0).replaceAllUsesWith(replacement);
    op.erase();
  });
}

void ReplaceMatmulPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
}

std::unique_ptr<mlir::Pass> createReplaceMatmulPass() {
  return std::make_unique<ReplaceMatmulPass>();
}


} // namespace analog
} // namespace mlir
