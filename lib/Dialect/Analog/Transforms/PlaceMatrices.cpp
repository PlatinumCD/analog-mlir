#include "analog-mlir/Dialect/Analog/Transforms/PlaceMatrices.h"
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
//   PlaceMatricesPass - Pass
// =====--------------------------------=====

llvm::StringRef PlaceMatricesPass::getArgument() const {
  return "analog-place-matrices";
}

llvm::StringRef PlaceMatricesPass::getDescription() const {
  return "Generate array placement loops that emit analog.array.matrix.place for each array-grid coordinate";
}

void PlaceMatricesPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::MatrixPartitionOp op) {

    auto grid = op.getResult();
    auto gridTy = llvm::dyn_cast<analog::MatrixGridType>(grid.getType());
    if (!gridTy) {
      return;
    }

    auto gridShape = gridTy.getGridShape();
    int64_t numArrayRows = gridShape[0]; 
    int64_t numArrayCols = gridShape[1]; 

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto loc = op.getLoc();

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value ubTr = builder.create<arith::ConstantIndexOp>(loc, numArrayRows);
    Value ubTc = builder.create<arith::ConstantIndexOp>(loc, numArrayCols);

    builder.create<scf::ForOp>(
      loc, zero, ubTr, one, ValueRange{},
      [&](OpBuilder &b1, Location loc, Value tr, ValueRange) {

        b1.create<scf::ForOp>(
          loc, zero, ubTc, one, ValueRange{},
          [&](OpBuilder &b2, Location loc, Value tc, ValueRange) {

            b2.create<analog::ArrayMatrixPlaceOp>(
              loc,
              grid,
              tr,
              tc,
              ValueRange{tr, tc}
            );

            b2.create<scf::YieldOp>(loc);
          });

        b1.create<scf::YieldOp>(loc);
      });

  });
}

void PlaceMatricesPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createPlaceMatricesPass() {
  return std::make_unique<PlaceMatricesPass>();
}


} // namespace analog
} // namespace mlir
