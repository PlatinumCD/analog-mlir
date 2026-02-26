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

// =====--------------------------------=====
//   PlaceVectorsPass - Pass
// =====--------------------------------=====

llvm::StringRef PlaceVectorsPass::getArgument() const {
  return "analog-place-vectors";
}

llvm::StringRef PlaceVectorsPass::getDescription() const {
  return "Generate varray placement loops that emit analog.array.vector.place for each vector array coordinate";
}

void PlaceVectorsPass::runOnOperation() {
  auto func = getOperation();

  func.walk([&](analog::VectorPartitionOp op) {
    auto slice = op.getResult();
    auto sliceTy = llvm::dyn_cast<analog::VectorSliceType>(slice.getType());
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

void PlaceVectorsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<analog::AnalogDialect>();
}

std::unique_ptr<mlir::Pass> createPlaceVectorsPass() {
  return std::make_unique<PlaceVectorsPass>();
}

} // namespace analog
} // namespace mlir
