#include "analog-mlir/Dialect/Analog/Transforms/PlaceVTiles.h"
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
//   PlaceVTilesPass - Pass
// =====--------------------------------=====

llvm::StringRef PlaceVTilesPass::getArgument() const {
  return "analog-place-vtiles";
}

llvm::StringRef PlaceVTilesPass::getDescription() const {
  return "TODO";
}

void PlaceVTilesPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::VTilePartitionOp op) {

    auto slice = op.getResult();
    auto sliceTy = llvm::dyn_cast<analog::VTileSliceType>(slice.getType());
    if (!sliceTy) {
      return;
    }

    Operation *next = op->getNextNode();
    if (!next) {
      return;
    }

    auto matmulOp = llvm::dyn_cast<linalg::MatmulOp>(next);
    if (!matmulOp) {
      return;
    }

    Value matrixTransposeInput = matmulOp.getInputs()[1]; 
    auto matrixTansposeInputTy = llvm::dyn_cast<RankedTensorType>(matrixTransposeInput.getType());
    if (!matrixTansposeInputTy) {
      return;
    }

    auto matrixTransposeShape = matrixTansposeInputTy.getShape();
    int64_t matrixRows = matrixTransposeShape[1];
    int64_t matrixCols = matrixTransposeShape[0];

    auto tileShape = sliceTy.getTileShape();
    int64_t tileRows = tileShape[0];
    int64_t tileCols = tileShape[1];

    int64_t numTileRows = (matrixRows + tileRows - 1) / tileRows;
    int64_t numTileCols = (matrixCols + tileCols - 1) / tileCols;

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto loc = op.getLoc();

    // Create index constants
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value ubTr = builder.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value ubTc = builder.create<arith::ConstantIndexOp>(loc, numTileCols);

    // Now build the loops
    builder.create<scf::ForOp>(
      loc, zero, ubTr, one, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value tr, ValueRange) {

        b1.create<scf::ForOp>(
          loc, zero, ubTc, one, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value tc, ValueRange) {

            b2.create<analog::VTilePlaceOp>(
              loc,
              slice,
              ValueRange{tr, tc}
            );

            b2.create<scf::YieldOp>(loc);
          });

        b1.create<scf::YieldOp>(loc);
      });
  });
}

void PlaceVTilesPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createPlaceVTilesPass() {
  return std::make_unique<PlaceVTilesPass>();
}


} // namespace analog
} // namespace mlir
