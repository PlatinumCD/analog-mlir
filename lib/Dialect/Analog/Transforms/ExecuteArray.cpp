#include "analog-mlir/Dialect/Analog/Transforms/ExecuteArray.h"
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
//   ExecuteArrayPass - Pass
// =====--------------------------------=====

llvm::StringRef ExecuteArrayPass::getArgument() const {
  return "analog-execute-array";
}

llvm::StringRef ExecuteArrayPass::getDescription() const {
  return "Insert ExecuteArray ops";
}

void ExecuteArrayPass::runOnOperation() {
  auto func = getOperation();

  std::deque<analog::MatrixGridType> gridQueue;

  // Find array partition
  func.walk([&](analog::MatrixPartitionOp op) {
    auto grid = op.getResult();
    auto gridTy = llvm::dyn_cast<analog::MatrixGridType>(grid.getType());
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

    auto arrayShape = gridTy.getArrayShape();
    int64_t arrayRows = arrayShape[0];

    auto gridShape = gridTy.getGridShape();
    int64_t numArrayRows = gridShape[0]; 
    int64_t numArrayCols = gridShape[1]; 

    auto f32Ty = builder.getF32Type();

    // memref<array_row x array_col x lanes x f32>
    auto arrayOutputBuffersTy = mlir::MemRefType::get({numArrayRows, numArrayCols, arrayRows}, f32Ty);
    Value arrayOutputBuffers = builder.create<memref::AllocOp>(loc, arrayOutputBuffersTy);

    auto arrayOutputBuffersOp = arrayOutputBuffers.getDefiningOp<memref::AllocOp>();
    arrayOutputBuffersOp->setAttr(
      "analog-alloc-id",
      builder.getI64IntegerAttr(alloc_id)
    );
    alloc_id++;

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);

    Value ubArrayRows = builder.create<arith::ConstantIndexOp>(loc, numArrayRows);
    Value ubArrayCols = builder.create<arith::ConstantIndexOp>(loc, numArrayCols);

    // for array-row
    builder.create<scf::ForOp>(
      loc, zero, ubArrayRows, one, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value tr, ValueRange) {

        // for array-col
        b1.create<scf::ForOp>(
          loc, zero, ubArrayCols, one, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value tc, ValueRange) {

            // Execute array
            Value array = b2.create<analog::ArrayExecuteOp>(loc, gridTy, ValueRange{tr, tc});

            // Store 1x(arrayRows) lanes into [tr, tc, :]
            b2.create<analog::ArrayStoreOp>(
              loc,
              array,
              arrayOutputBuffers,
              ValueRange{tr, tc}
            );

            b2.create<scf::YieldOp>(loc);
          });

        b1.create<scf::YieldOp>(loc);
      });
  });
}

void ExecuteArrayPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createExecuteArrayPass() {
  return std::make_unique<ExecuteArrayPass>();
}


} // namespace analog
} // namespace mlir
