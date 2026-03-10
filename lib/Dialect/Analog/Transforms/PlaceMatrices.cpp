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

namespace {


// Returns the grid type only for values already partitioned into
// analog matrix grids.

analog::MatrixGridType getPlacableMatrixGridType(Value value) {
  return llvm::dyn_cast<analog::MatrixGridType>(value.getType());
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef PlaceMatricesPass::getArgument() const {
  return "analog-place-matrices";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef PlaceMatricesPass::getDescription() const {
  return "Generate array placement loops that emit analog.array.matrix.place for each array-grid coordinate";
}


// Emits placement loops for each partitioned matrix so every array-grid
// coordinate receives a concrete placement op.

void PlaceMatricesPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::MatrixPartitionOp op) {
    auto grid = op.getResult();
    analog::MatrixGridType gridTy = getPlacableMatrixGridType(grid);
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


// Declares the analog dialect required for the placement ops inserted
// by this pass.

void PlaceMatricesPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createPlaceMatricesPass() {
  return std::make_unique<PlaceMatricesPass>();
}

} // namespace analog
} // namespace mlir
