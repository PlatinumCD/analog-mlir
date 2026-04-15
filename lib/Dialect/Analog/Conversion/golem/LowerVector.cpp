#include "analog-mlir/Dialect/Analog/Conversion/golem/GolemUtils.h"

namespace {

// Erases vector materialization once the type converter carries vector storage.
class VectorFromTensorLowering
    : public mlir::OpConversionPattern<mlir::analog::VectorFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Replaces the analog wrapper with the already-converted tensor input.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::VectorFromTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

// Erases slice partition views after their tiling metadata has been consumed.
class VectorPartitionLowering
    : public mlir::OpConversionPattern<mlir::analog::VectorPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Keeps the lowered vector value flowing where the slice wrapper was used.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::VectorPartitionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getVector());
    return mlir::success();
  }
};

// Lowers one vector-slice placement into scratch preparation and a load call.
class ArrayVectorPlaceLowering
    : public mlir::OpConversionPattern<mlir::analog::ArrayVectorPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Materializes the selected slice for the runtime and removes the place op.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::ArrayVectorPlaceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Validate both the Analog slice contract and the lowered vector shape.
    auto sliceType =
        llvm::dyn_cast<mlir::analog::VectorSliceType>(op.getInput().getType());
    if (!sliceType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected analog.vector.slice input type");
    }

    auto vectorType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!vectorType || vectorType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected lowered vector tensor<1xn>");
    }

    // Build dynamic offsets and clamped copy bounds for the selected slice.
    mlir::Value fullMemref = mlir::analog::golem::materializeTensorMemref(
        rewriter, op.getLoc(), adaptor.getInput());
    auto plan = mlir::analog::golem::buildVectorPlacementPlan(
        rewriter, op, adaptor.getSliceIndex(), fullMemref, sliceType);

    // Prepare a zero-padded scratch row before copying the live source region.
    auto maybeScratch = mlir::analog::golem::allocateZeroedScratchTile(
        rewriter, op, {1, plan.arrayCols}, vectorType.getElementType(), plan.c1,
        plan.cArrayCols, plan.c0, plan.c1);
    if (mlir::failed(maybeScratch))
      return mlir::failure();

    mlir::Value arrayMemref = *maybeScratch;
    mlir::analog::golem::copyVectorSliceIntoScratch(
        rewriter, op.getLoc(), fullMemref, arrayMemref, plan.colOffset,
        plan.copyCols, plan.c0, plan.c1);

    // Prefer explicit hardware coordinates, falling back to the slice column.
    mlir::Value row = plan.c0;
    mlir::Value col = adaptor.getSliceIndex();
    if (adaptor.getIndices().size() >= 2) {
      row = adaptor.getIndices()[0];
      col = adaptor.getIndices()[1];
    }

    // Encode the grid coordinate expected by the runtime and emit the load call.
    mlir::Value arrayId = mlir::analog::golem::buildLinearArrayId(
        rewriter, op.getLoc(), row, col, plan.gridCols);
    mlir::analog::golem::emitIntrinsicCall(
        rewriter, op.getLoc(), mlir::analog::golem::kLoadIntrinsicName,
        {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace analog {
namespace golem {

// Registers vector wrapper erasure and placement patterns for Golem lowering.
void populateLowerVectorPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx) {
  patterns.add<VectorFromTensorLowering, VectorPartitionLowering,
               ArrayVectorPlaceLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace analog
} // namespace mlir
