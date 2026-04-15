#include "analog-mlir/Dialect/Analog/Conversion/golem/GolemUtils.h"

namespace {

// Erases matrix materialization once the type converter carries matrix storage.
class MatrixFromTensorLowering
    : public mlir::OpConversionPattern<mlir::analog::MatrixFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Replaces the analog wrapper with the already-converted tensor input.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::MatrixFromTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

// Erases partition views after their grid shape has been captured by the type.
class MatrixPartitionLowering
    : public mlir::OpConversionPattern<mlir::analog::MatrixPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Keeps the lowered matrix value flowing where the grid wrapper was used.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::MatrixPartitionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getMatrix());
    return mlir::success();
  }
};

// Lowers one matrix-grid placement into scratch preparation and a set call.
class ArrayMatrixPlaceLowering
    : public mlir::OpConversionPattern<mlir::analog::ArrayMatrixPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Materializes the selected tile for the runtime and removes the place op.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::ArrayMatrixPlaceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Validate both the Analog grid contract and the lowered matrix shape.
    auto gridType =
        llvm::dyn_cast<mlir::analog::MatrixGridType>(op.getInput().getType());
    if (!gridType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected analog.matrix.grid input type");
    }

    auto matrixType =
        llvm::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!matrixType || matrixType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected lowered matrix tensor<mxn>");
    }

    // Build dynamic offsets and clamped copy bounds for the selected array tile.
    mlir::Value fullMemref = mlir::analog::golem::materializeTensorMemref(
        rewriter, op.getLoc(), adaptor.getInput());
    auto plan = mlir::analog::golem::buildMatrixPlacementPlan(
        rewriter, op, adaptor.getRowIndex(), adaptor.getColIndex(), fullMemref,
        gridType);

    // Prepare a zero-padded scratch tile before copying the live source region.
    auto maybeScratch = mlir::analog::golem::allocateZeroedScratchTile(
        rewriter, op, {plan.arrayRows, plan.arrayCols},
        matrixType.getElementType(), plan.cArrayRows, plan.cArrayCols, plan.c0,
        plan.c1);
    if (mlir::failed(maybeScratch))
      return mlir::failure();

    mlir::Value arrayMemref = *maybeScratch;
    mlir::analog::golem::copyMatrixTileIntoScratch(
        rewriter, op.getLoc(), fullMemref, arrayMemref, plan.rowOffset,
        plan.colOffset, plan.copyRows, plan.copyCols, plan.c0, plan.c1);

    // Encode the grid coordinate expected by the runtime and emit the set call.
    mlir::Value arrayId = mlir::analog::golem::buildLinearArrayId(
        rewriter, op.getLoc(), adaptor.getRowIndex(), adaptor.getColIndex(),
        plan.gridCols);
    mlir::analog::golem::emitIntrinsicCall(
        rewriter, op.getLoc(), mlir::analog::golem::kSetIntrinsicName,
        {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace analog {
namespace golem {

// Registers matrix wrapper erasure and placement patterns for Golem lowering.
void populateLowerMatrixPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx) {
  patterns.add<MatrixFromTensorLowering, MatrixPartitionLowering,
               ArrayMatrixPlaceLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace analog
} // namespace mlir
