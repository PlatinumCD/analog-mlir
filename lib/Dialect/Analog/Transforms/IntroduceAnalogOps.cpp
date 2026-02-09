#include "analog-mlir/Dialect/Analog/Transforms/IntroduceAnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
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
//   IntroduceAnalogOpsPass - Pass
// =====--------------------------------=====

llvm::StringRef IntroduceAnalogOpsPass::getArgument() const {
  return "analog-introduce-ops";
}

llvm::StringRef IntroduceAnalogOpsPass::getDescription() const {
  return "Identify and transform linalg matmuls to analog operations.";
}

void IntroduceAnalogOpsPass::runOnOperation() {
  auto func = getOperation();


  func.walk([&](mlir::linalg::MatmulOp op) {

    llvm::errs() << op << "\n";

    // Get the transpose operation
    Value matmulRhs = op.getInputs()[1];
    auto transposeOp = matmulRhs.getDefiningOp<mlir::linalg::TransposeOp>();
    if (!transposeOp) {
      return;
    }
    llvm::errs() << transposeOp << "\n";


    Value transposeDest = transposeOp.getInit();
    auto emptyOp = transposeDest.getDefiningOp<mlir::tensor::EmptyOp>();
    if (!emptyOp) {
      return;
    }
    llvm::errs() << *emptyOp << "\n";
  
  });
}

void IntroduceAnalogOpsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createIntroduceAnalogOpsPass() {
  return std::make_unique<IntroduceAnalogOpsPass>();
}


} // namespace analog
} // namespace mlir
