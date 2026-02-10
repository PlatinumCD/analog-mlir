#include "analog-mlir/Dialect/Analog/Transforms/ExecuteTiles.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <deque>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Support/LLVM.h>


using namespace mlir;

namespace mlir {
namespace analog {

// =====--------------------------------=====
//   ExecuteTilesPass - Pass
// =====--------------------------------=====

llvm::StringRef ExecuteTilesPass::getArgument() const {
  return "analog-execute-tiles";
}

llvm::StringRef ExecuteTilesPass::getDescription() const {
  return "Insert ExecuteTiles ops";
}

void ExecuteTilesPass::runOnOperation() {
  auto func = getOperation();

  std::deque<analog::TileGridType> gridQueue;

  // Find tile partition
  func.walk([&](analog::TilePartitionOp op) {
    auto grid = op.getResult();
    auto gridTy = llvm::dyn_cast<analog::TileGridType>(grid.getType());
    if (!gridTy) {
      return;
    }

    gridQueue.push_back(gridTy);
  });


  int64_t alloc_id = 0;
  // Find matmul ops
  func.walk([&](mlir::linalg::MatmulOp op) {

    OpBuilder builder(op);
    builder.setInsertionPoint(op);
    auto loc = op.getLoc();

    auto gridTy = gridQueue.front();
    gridQueue.pop_front();

    auto tileShape = gridTy.getTileShape();
    int64_t tileRows = tileShape[0];

    auto gridShape = gridTy.getGridShape();
    int64_t numTileRows = gridShape[0]; 
    int64_t numTileCols = gridShape[1]; 

    auto f32Ty = builder.getF32Type();

    // memref<tile_row x tile_col x lanes x f32>
    auto tileOutputBuffersTy = mlir::MemRefType::get({numTileRows, numTileCols, tileRows}, f32Ty);
    Value tileOutputBuffers = builder.create<memref::AllocOp>(loc, tileOutputBuffersTy);

    auto tileOutputBuffersOp = tileOutputBuffers.getDefiningOp<memref::AllocOp>();
    tileOutputBuffersOp->setAttr(
      "analog-alloc-id",
      builder.getI64IntegerAttr(alloc_id)
    );
    alloc_id++;

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);

    Value ubTileRows = builder.create<arith::ConstantIndexOp>(loc, numTileRows);
    Value ubTileCols = builder.create<arith::ConstantIndexOp>(loc, numTileCols);

    // for tile-row
    builder.create<scf::ForOp>(
      loc, zero, ubTileRows, one, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value tr, ValueRange) {

        // for tile-col
        b1.create<scf::ForOp>(
          loc, zero, ubTileCols, one, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value tc, ValueRange) {

            // Execute tile
            Value tile = b2.create<analog::TileExecuteOp>(loc, gridTy, ValueRange{tr, tc});

            // Store 1x(tileRows) lanes into [tr, tc, :]
            b2.create<analog::TileStoreOp>(
              loc,
              tile,
              tileOutputBuffers,
              ValueRange{tr, tc}
            );

            b2.create<scf::YieldOp>(loc);
          });

        b1.create<scf::YieldOp>(loc);
      });
  });
}

void ExecuteTilesPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createExecuteTilesPass() {
  return std::make_unique<ExecuteTilesPass>();
}


} // namespace analog
} // namespace mlir
