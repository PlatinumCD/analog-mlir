#include "analog-mlir/Dialect/Analog/Transforms/CombineTileResults.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
//   CombineTileResultsPass - Pass
// =====--------------------------------=====

llvm::StringRef CombineTileResultsPass::getArgument() const {
  return "analog-combine-tile-results";
}

llvm::StringRef CombineTileResultsPass::getDescription() const {
  return "Insert CombineTileResults ops";
}

void CombineTileResultsPass::runOnOperation() {
  auto func = getOperation();

  std::deque<analog::TileGridType> gridQueue;
  std::deque<mlir::Value> memrefValueQueue;

  // Find tile partition
  func.walk([&](analog::TilePartitionOp op) {
    auto grid = op.getResult();
    auto gridTy = llvm::dyn_cast<analog::TileGridType>(grid.getType());
    if (!gridTy) {
      return;
    }
    gridQueue.push_back(gridTy);
  });

  // Find AllocOp
  func.walk([&](memref::AllocOp op) {
    if (op->getAttr("analog-alloc-id")) {
      auto ref = op.getResult();
      memrefValueQueue.push_back(ref);
    }
  });

  // Find matmul
  func.walk([&](mlir::linalg::MatmulOp op) {
    OpBuilder builder(op);
    builder.setInsertionPoint(op);
    auto loc = op.getLoc();

    // Grid data
    auto gridTy = gridQueue.front();
    gridQueue.pop_front();

    auto gridShape = gridTy.getGridShape();
    int64_t gridRows = gridShape[0];
    int64_t gridCols = gridShape[1];

    // Memref data
    auto memrefVal = memrefValueQueue.front();
    memrefValueQueue.pop_front();

    auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(memrefVal.getType());
    if (!memrefTy) {
      return;
    }

    auto memrefShape = memrefTy.getShape();
    int64_t memTileLane = memrefShape[2];

    // ================================================================
    // Assumptions / Shape bindings (from your existing variables)
    // ================================================================
    // gridRows, gridCols        : gridTy.getGridShape()
    // tileRows, tileCols        : tileTy.getTileShape()
    // memTileRows, memTileCols  : memrefTy.getShape()[0,1]
    // memTileLane               : memrefTy.getShape()[2]
    // tileBufs                  : memref<gridRows*gridCols x memTileRows x memTileLane x f32>
    // Result shape              : <1 x (gridRows * memTileLane) x f32>
    // Reduction                 : column-wise reduction of tiles into row buffers

    // ================================================================
    // Constants
    // ================================================================
    auto f32Ty   = builder.getF32Type();

    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

    Value cGridRows = builder.create<arith::ConstantIndexOp>(loc, gridRows);
    Value cGridCols = builder.create<arith::ConstantIndexOp>(loc, gridCols);
    Value cLane     = builder.create<arith::ConstantIndexOp>(loc, memTileLane);

    Value c0f = builder.create<arith::ConstantFloatOp>(loc, f32Ty, llvm::APFloat(0.0f));


    // ================================================================
    // Row-wise reduction buffers: gridRows  x lane
    // ================================================================
    auto rowBufTy = MemRefType::get({gridRows, memTileLane}, f32Ty);
    Value rowBufs = builder.create<memref::AllocOp>(loc, rowBufTy);

    // ================================================================
    // Initialize row buffers to zero
    // ================================================================
    builder.create<scf::ForOp>(
      loc, c0, cGridRows, c1, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value r, ValueRange) {

        b1.create<scf::ForOp>(
          loc, c0, cLane, c1, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value j, ValueRange) {

            b2.create<memref::StoreOp>(
              loc, c0f, rowBufs, ValueRange{r, j});

            b2.create<scf::YieldOp>(loc);

          });

        b1.create<scf::YieldOp>(loc);
      });

    // ================================================================
    // Reduce column tiles into row tiles
    // tileId = r * gridCols + c
    // ================================================================
    builder.create<scf::ForOp>(
      loc, c0, cGridRows, c1, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value r, ValueRange) {

        b1.create<scf::ForOp>(
          loc, c0, cGridCols, c1, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value c, ValueRange) {

            b2.create<scf::ForOp>(
              loc, c0, cLane, c1, ValueRange{},
              [&](OpBuilder &b3, Location loc, Value j, ValueRange) {

                // Get temporary row buffers
                Value acc = b3.create<memref::LoadOp>(loc, rowBufs, ValueRange{r, j});

                // Get tile results
                Value val = b3.create<memref::LoadOp>(loc, memrefVal, ValueRange{r, c, j});

                // temporary row buffer += val
                Value sum = b3.create<arith::AddFOp>(loc, acc, val);
                b3.create<memref::StoreOp>(loc, sum, rowBufs, ValueRange{r, j});

                b3.create<scf::YieldOp>(loc);
              });
            
              b2.create<scf::YieldOp>(loc);
          });

          b1.create<scf::YieldOp>(loc);
      });

    // ================================================================
    // Assemble final output vector: 1 x (gridRows * lane)
    // col = r * lane + j
    // ================================================================
    int64_t outCols = gridRows * memTileLane;
    auto outTy = MemRefType::get({1, outCols}, f32Ty);
    Value out = builder.create<memref::AllocOp>(loc, outTy);

    builder.create<scf::ForOp>(
      loc, c0, cGridRows, c1, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value r, ValueRange) {

        b1.create<scf::ForOp>(
          loc, c0, cLane, c1, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value j, ValueRange) {

            // Load value from temporary buffer
            Value v = b2.create<memref::LoadOp>(loc, rowBufs, ValueRange{r, j});

            // compute index
            Value colOffset = b2.create<arith::MulIOp>(loc, r, cLane);
            Value col = b2.create<arith::AddIOp>(loc, colOffset, j);

            // store temporary value into output buffer
            b2.create<memref::StoreOp>(loc, v, out, ValueRange{c0, col});
              
            b2.create<scf::YieldOp>(loc);
          });

        b1.create<scf::YieldOp>(loc);
      });

    // ================================================================
    // Materialize tensor result
    // ================================================================
    auto resultTy = RankedTensorType::get({1, outCols}, f32Ty);
    auto toTensor = builder.create<bufferization::ToTensorOp>(loc, resultTy, out);
    toTensor->setAttr("restrict", builder.getUnitAttr());
  });
}

void CombineTileResultsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
}

std::unique_ptr<mlir::Pass> createCombineTileResultsPass() {
  return std::make_unique<CombineTileResultsPass>();
}


} // namespace analog
} // namespace mlir
