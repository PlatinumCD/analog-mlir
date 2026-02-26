#include "analog-mlir/Dialect/Analog/Transforms/PlaceMatrices.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/Transforms/TransformUtils.h"

#include "llvm/Support/Casting.h"
#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>

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

    detail::build2DIndexLoopNest(
      builder, loc, numArrayRows, numArrayCols,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value tr, Value tc) {
        loopBuilder.create<analog::ArrayMatrixPlaceOp>(
          loopLoc,
          grid,
          tr,
          tc,
          ValueRange{tr, tc}
        );
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
