#include "analog-mlir/Dialect/Analog/Transforms/MaterializeVectorFromTensor.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>


using namespace mlir;

namespace mlir {
namespace analog {

namespace {


// Returns the ranked tensor type only for rank-2 floating-point tensor
// values that can become analog vectors.

RankedTensorType getMaterializableVectorTensorType(Value value) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy || tensorTy.getRank() != 2) {
    return {};
  }
  if (!llvm::isa<FloatType>(tensorTy.getElementType())) {
    return {};
  }

  return tensorTy;
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef MaterializeVectorFromTensorPass::getArgument() const {
  return "analog-materialize-vector";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef MaterializeVectorFromTensorPass::getDescription() const {
  return "Transform dense resources into analog vector types";
}


// Materializes each eligible matmul input tensor as an analog vector
// once and inserts the conversion immediately before the matmul.

void MaterializeVectorFromTensorPass::runOnOperation() {
  auto func = getOperation();
  llvm::DenseSet<Value> materializedInputs;

  func.walk([&](mlir::linalg::MatmulOp op) {
    Value inputVector = op.getInputs()[0];
    RankedTensorType inputVectorTy =
        getMaterializableVectorTensorType(inputVector);
    if (!inputVectorTy) {
      return;
    }

    if (!materializedInputs.insert(inputVector).second) {
      return;
    }

    OpBuilder builder(op);
    builder.setInsertionPoint(op);

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


// Declares the analog dialect required for the vector materialization
// op inserted by this pass.

void MaterializeVectorFromTensorPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createMaterializeVectorFromTensorPass() {
  return std::make_unique<MaterializeVectorFromTensorPass>();
}


} // namespace analog
} // namespace mlir
