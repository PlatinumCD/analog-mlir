#include "analog-mlir/Dialect/Analog/Transforms/PlaceVectors.h"
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


// Returns the slice type only for values already partitioned into
// analog vector slices.

analog::VectorSliceType getPlacableVectorSliceType(Value value) {
  return llvm::dyn_cast<analog::VectorSliceType>(value.getType());
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef PlaceVectorsPass::getArgument() const {
  return "analog-place-vectors";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef PlaceVectorsPass::getDescription() const {
  return "Generate varray placement loops that emit analog.array.vector.place for each vector array coordinate";
}


// Emits placement loops for each partitioned vector so every array-grid
// coordinate receives a concrete placement op.

void PlaceVectorsPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::VectorPartitionOp op) {
    auto slice = op.getResult();
    analog::VectorSliceType sliceTy = getPlacableVectorSliceType(slice);
    if (!sliceTy) {
      return;
    }

    auto gridShape = sliceTy.getGridShape();
    int64_t numArrayRows = gridShape[0]; 
    int64_t numArrayCols = gridShape[1];

    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    auto loc = op.getLoc();

    detail::build2DIndexLoopNest(
      builder, loc, numArrayRows, numArrayCols,
      [&](OpBuilder &loopBuilder, Location loopLoc, Value tr, Value tc) {
        loopBuilder.create<analog::ArrayVectorPlaceOp>(
          loopLoc,
          slice,
          tc,
          ValueRange{tr, tc}
        );
      });
  });
}


// Declares the analog dialect required for the placement ops inserted
// by this pass.

void PlaceVectorsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createPlaceVectorsPass() {
  return std::make_unique<PlaceVectorsPass>();
}

} // namespace analog
} // namespace mlir
