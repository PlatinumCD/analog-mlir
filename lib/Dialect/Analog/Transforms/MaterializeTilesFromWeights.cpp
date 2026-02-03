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
#include <algorithm>
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

    Value input = op.getInput();
    auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensorType)
      return;

    int64_t tileRows   = tile_rows;
    int64_t tileCols   = tile_cols;
    int64_t matrixRows = tensorType.getShape()[0];
    int64_t matrixCols = tensorType.getShape()[1];

    int64_t numTileRows = (matrixRows + tileRows - 1) / tileRows;
    int64_t numTileCols = (matrixCols + tileCols - 1) / tileCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    auto layerAttr = op.getLayerAttr();
    int64_t tilesReq = 0;

    for (int64_t tileRowIdx = 0; tileRowIdx < numTileRows; ++tileRowIdx) {
      for (int64_t tileColIdx = 0; tileColIdx < numTileCols; ++tileColIdx) {

        int64_t rowOffset = tileRowIdx * tileRows;
        int64_t colOffset = tileColIdx * tileCols;

        int64_t tileNumRows = std::min(tileRows, matrixRows - rowOffset);
        int64_t tileNumCols = std::min(tileCols, matrixCols - colOffset);

        // Build results type
        auto tileType = analog::TileType::get(
            builder.getContext(),
            {tileNumRows, tileNumCols},
            tensorType.getElementType(),
            /*stride=*/0,
            /*base=*/0
        );

        // Build TilesFromWeights Operation
        builder.create<analog::TilesFromWeightsOp>(
            op.getLoc(),
            tileType,
            op.getResult(),
            builder.getI64IntegerAttr(tileNumRows),
            builder.getI64IntegerAttr(tileNumCols),
            builder.getI64IntegerAttr(tileRowIdx),
            builder.getI64IntegerAttr(tileColIdx),
            layerAttr
        );

        ++tilesReq;
      }
    }

    // Update weights with number of tiles needed
    op->setAttr("tiles_req", builder.getI64IntegerAttr(tilesReq));
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
