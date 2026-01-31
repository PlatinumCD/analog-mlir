#include "analog-mlir/Transforms/NN/TagNNLayerIndex.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace mlir {
namespace nn {

llvm::StringRef TagNNLayerIndexPass::getArgument() const {
  return "nn-tag-layer-index";
}

llvm::StringRef TagNNLayerIndexPass::getDescription() const {
  return "Assign nn.layer_index attributes to dense tensor constants";
}

void TagNNLayerIndexPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = func.getContext();

  int64_t layer = 0;

  func.walk([&](arith::ConstantOp cst) {
    auto tensorTy = llvm::dyn_cast<RankedTensorType>(cst.getType());
    if (!tensorTy)
      return;

    if (!llvm::isa<DenseResourceElementsAttr>(cst.getValue()))
      return;

    cst->setAttr(
        "nn.layer_index",
        IntegerAttr::get(IntegerType::get(ctx, 64), layer));

    ++layer;
  });
}

std::unique_ptr<mlir::Pass> createTagNNLayerIndexPass() {
  return std::make_unique<TagNNLayerIndexPass>();
}

} // namespace nn
} // namespace mlir

