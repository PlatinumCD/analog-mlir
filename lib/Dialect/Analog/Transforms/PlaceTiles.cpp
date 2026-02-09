#include "analog-mlir/Dialect/Analog/Transforms/PlaceTiles.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
//   PlaceTilesPass - Pass
// =====--------------------------------=====

llvm::StringRef PlaceTilesPass::getArgument() const {
  return "analog-place-tiles";
}

llvm::StringRef PlaceTilesPass::getDescription() const {
  return "TODO";
}

void PlaceTilesPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::TilePartitionOp op) {

    auto grid = op.getResult();
    auto gridTy = llvm::dyn_cast<analog::TileGridType>(grid.getType());
    if (!gridTy) {
      return;
    }

    auto matrixTy = gridTy.getMatrix(); 
    auto matrixShape = matrixTy.getShape();
    int64_t matrixRows = matrixShape[0];
    int64_t matrixCols = matrixShape[1];

    auto tileShape = gridTy.getTileShape();
    int64_t tileRows = tileShape[0];
    int64_t tileCols = tileShape[1];

    int64_t numTileRows = (matrixRows + tileRows - 1) / tileRows;
    int64_t numTileCols = (matrixCols + tileCols - 1) / tileCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto loc = op.getLoc();

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value ubTr = builder.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value ubTc = builder.create<arith::ConstantIndexOp>(loc, numTileCols);

    builder.create<scf::ForOp>(
      loc, zero, ubTr, one, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value tr, ValueRange) {

        b1.create<scf::ForOp>(
          loc, zero, ubTc, one, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value tc, ValueRange) {

            b2.create<analog::TilePlaceOp>(
              loc,
              grid,
              ValueRange{tr, tc}
            );

            b2.create<scf::YieldOp>(loc);
          });

        b1.create<scf::YieldOp>(loc);
      });

  });
}

void PlaceTilesPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createPlaceTilesPass() {
  return std::make_unique<PlaceTilesPass>();
}


} // namespace analog
} // namespace mlir
