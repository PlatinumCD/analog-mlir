#include "analog-mlir/Dialect/Analog/Transforms/PrepareConv2DToMatmul.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

constexpr StringLiteral kDeleteInFuturePassAttr = "analog.delete_in_future_pass";


// Returns the ranked tensor type only for static rank-4 tensor
// constants that this pass knows how to flatten.

RankedTensorType getFlattenableTensorType(arith::ConstantOp op) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(op.getType());
  if (!tensorTy || tensorTy.getRank() != 4 || !tensorTy.hasStaticShape()) {
    return {};
  }

  return tensorTy;
}


// Computes the rank-2 tensor type produced by flattening the spatial
// and channel dimensions of a conv weight tensor.

RankedTensorType buildFlattenedTensorType(RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();
  int64_t flattenedCols = shape[1] * shape[2] * shape[3];
  return RankedTensorType::get(
      {shape[0], flattenedCols}, tensorTy.getElementType());
}


// Rebuilds the constant payload with the flattened type while
// preserving either dense or resource-backed storage.

TypedAttr buildFlattenedAttr(arith::ConstantOp op, RankedTensorType flattenedTy) {
  if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(op.getValue())) {
    return denseAttr.reshape(flattenedTy);
  }

  if (auto resourceAttr =
          llvm::dyn_cast<DenseResourceElementsAttr>(op.getValue())) {
    return DenseResourceElementsAttr::get(
        flattenedTy, resourceAttr.getRawHandle());
  }

  return {};
}


// Inserts the flattened constant after the original one and marks the
// old op for deletion by a later cleanup pass.

void replaceWithFlattenedConstant(arith::ConstantOp op,
                                  RankedTensorType flattenedTy,
                                  TypedAttr flattenedAttr) {
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  builder.create<arith::ConstantOp>(op.getLoc(), flattenedTy, flattenedAttr);
  op->setAttr(kDeleteInFuturePassAttr, builder.getUnitAttr());
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef PrepareConv2DToMatmulPass::getArgument() const {
  return "analog-prepare-conv2d-to-matmul";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef PrepareConv2DToMatmulPass::getDescription() const {
  return "Flatten rank-4 tensor constants into rank-2 tensors for later conv-to-matmul lowering";
}


// Rewrites rank-4 tensor constants into flattened rank-2 forms so
// later passes can treat conv weights like matmul operands.

void PrepareConv2DToMatmulPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](arith::ConstantOp op) {
    RankedTensorType tensorTy = getFlattenableTensorType(op);
    if (!tensorTy) {
      return;
    }

    RankedTensorType flattenedTy = buildFlattenedTensorType(tensorTy);
    TypedAttr flattenedAttr = buildFlattenedAttr(op, flattenedTy);
    if (!flattenedAttr) {
      return;
    }

    replaceWithFlattenedConstant(op, flattenedTy, flattenedAttr);
  });
}


// Declares the dialects this pass may create while rewriting
// constants.

void PrepareConv2DToMatmulPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createPrepareConv2DToMatmulPass() {
  return std::make_unique<PrepareConv2DToMatmulPass>();
}

} // namespace analog
} // namespace mlir
