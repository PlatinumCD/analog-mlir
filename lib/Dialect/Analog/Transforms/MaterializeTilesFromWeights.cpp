#include "analog-mlir/Dialect/Analog/Transforms/MaterializeTilesFromWeights.h"
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
//   MaterializeTilesFromWeightsPass - Pass
// =====--------------------------------=====

llvm::StringRef MaterializeTilesFromWeightsPass::getArgument() const {
  return "analog-materialize-tiles";
}

llvm::StringRef MaterializeTilesFromWeightsPass::getDescription() const {
  return "Transform into analog weight types into analog tile types";
}

void MaterializeTilesFromWeightsPass::runOnOperation() {
  auto func = getOperation();


  func.walk([&](analog::WeightsFromConstOp op) {

//    if (op->hasAttr("analog.tiles.materialized")) {
//      return;
//    }

    Value input = op.getInput();
    auto tensorTy = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensorTy) {
        return;
    }

    int64_t matrix_rows = tensorTy.getShape()[0];
    int64_t matrix_cols = tensorTy.getShape()[1];
    llvm::errs() << "matrix rows=" << matrix_rows << ", ";
    llvm::errs() << "matrix cols=" << matrix_cols << ", ";
    llvm::errs() << "tile rows=" << tile_rows << ", ";
    llvm::errs() << "tile cols=" << tile_cols << "\n";


    if ((tile_rows > matrix_rows) && (tile_cols > matrix_cols)) {

      // Set insertion point
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);

      // Build results type
      auto tileTy = analog::TileType::get(
         builder.getContext(), 
         tensorTy.getShape(),
         tensorTy.getElementType(),
         /*stride=*/0,
         /*base=*/0
      );

      auto tileRowAttr = builder.getI64IntegerAttr(tile_rows);
      auto tileColAttr = builder.getI64IntegerAttr(tile_cols);
      auto layerAttr = op.getLayerAttr();

      builder.create<analog::TilesFromWeightsOp>(
        op.getLoc(),
        tileTy,
        op.getResult(),
        tileRowAttr,
        tileColAttr,
        layerAttr
      );


    } else {
      // TODO: tiling operations here
      return;
    }


    // Build result type
//    auto weightsTy = analog::WeightsType::get(
//       builder.getContext(),
//        tensorTy.getShape(),
//        tensorTy.getElementType()
//    );

//    auto layerAttr = builder.getI64IntegerAttr(layer_index);

//    builder.create<analog::WeightsFromConstOp>(
//        op.getLoc(),
//        weightsTy,
//        op.getResult(),
//        layerAttr
//    );
//
//    op->setAttr("analog.tiles.materialized", builder.getUnitAttr());

  });
}

void MaterializeTilesFromWeightsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createMaterializeTilesFromWeightsPass() {
  return std::make_unique<MaterializeTilesFromWeightsPass>();
}


} // namespace analog
} // namespace mlir
