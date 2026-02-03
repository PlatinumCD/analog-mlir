#include "analog-mlir/Dialect/Analog/Transforms/MaterializeWeightsFromConst.h"
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
//   MaterializeWeightsFromConstPass - Pass
// =====--------------------------------=====

llvm::StringRef MaterializeWeightsFromConstPass::getArgument() const {
  return "analog-materialize-weights";
}

llvm::StringRef MaterializeWeightsFromConstPass::getDescription() const {
  return "Transform dense resources into analog weight types";
}

void MaterializeWeightsFromConstPass::runOnOperation() {
  auto func = getOperation();

  uint64_t layer_index = 0;

  func.walk([&](arith::ConstantOp op) {

    auto tensorTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!tensorTy) {
      return;
    }

    if (!llvm::isa<DenseResourceElementsAttr>(op.getValue())) {
      return;
    }

    if (op->hasAttr("layer")) {
      return;
    }

    OpBuilder builder(op);
    
    // Set insertion point
    builder.setInsertionPointAfter(op);

    // Build result type
    auto weightsTy = analog::WeightsType::get(
        builder.getContext(),
        tensorTy.getShape(),
        tensorTy.getElementType()
    );

    auto layerAttr = builder.getI64IntegerAttr(layer_index);

    builder.create<analog::WeightsFromConstOp>(
        op.getLoc(),
        weightsTy,
        op.getResult(),
        layerAttr
    );

    op->setAttr("layer", layerAttr);

    layer_index++;
  });
}

void MaterializeWeightsFromConstPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createMaterializeWeightsFromConstPass() {
  return std::make_unique<MaterializeWeightsFromConstPass>();
}


} // namespace analog
} // namespace mlir

