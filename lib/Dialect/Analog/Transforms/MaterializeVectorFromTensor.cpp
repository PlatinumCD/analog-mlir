#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
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
//   MaterializeVectorFromTensorPass - Pass
// =====--------------------------------=====

llvm::StringRef MaterializeVectorFromTensorPass::getArgument() const {
  return "analog-materialize-vector";
}

llvm::StringRef MaterializeVectorFromTensorPass::getDescription() const {
  return "Transform dense resources into analog vector types";
}

void MaterializeVectorFromTensorPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](mlir::linalg::MatmulOp op) {

    OpBuilder builder(op);
    builder.setInsertionPoint(op);

    Value inputVector = op.getInputs()[0];
    auto inputVectorTy = llvm::dyn_cast<RankedTensorType>(inputVector.getType());
    if (!inputVectorTy) {
      return;
    }

    auto vectorTy = analog::VectorType::get(
      builder.getContext(),
      inputVectorTy.getShape(),
      inputVectorTy.getElementType()
    );

    builder.create<analog::VectorFromTensorOp>(
      op.getLoc(),
      vectorTy,
      inputVector
    );
  });
}

void MaterializeVectorFromTensorPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createMaterializeVectorFromTensorPass() {
  return std::make_unique<MaterializeVectorFromTensorPass>();
}


} // namespace analog
} // namespace mlir

