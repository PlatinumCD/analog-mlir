#include "analog-mlir/Dialect/Analog/Transforms/MaterializeMatrixFromTensor.h"
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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>


using namespace mlir;

namespace mlir {
namespace analog {

// =====--------------------------------=====
//   MaterializeMatrixFromTensorPass - Pass
// =====--------------------------------=====

llvm::StringRef MaterializeMatrixFromTensorPass::getArgument() const {
  return "analog-materialize-matrix";
}

llvm::StringRef MaterializeMatrixFromTensorPass::getDescription() const {
  return "Transform dense resources into analog matrix types";
}

void MaterializeMatrixFromTensorPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](arith::ConstantOp op) {

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!tensorTy) {
      return;
    }

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    // Build result type
    auto matrixTy = analog::MatrixType::get(
        builder.getContext(),
        tensorTy.getShape(),
        tensorTy.getElementType()
    );

    builder.create<analog::MatrixFromTensorOp>(
        op.getLoc(),
        matrixTy,
        op.getResult()
    );
  });
}

void MaterializeMatrixFromTensorPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createMaterializeMatrixFromTensorPass() {
  return std::make_unique<MaterializeMatrixFromTensorPass>();
}


} // namespace analog
} // namespace mlir

