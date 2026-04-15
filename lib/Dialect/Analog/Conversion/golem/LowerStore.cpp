#include "analog-mlir/Dialect/Analog/Conversion/golem/GolemUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SmallVector.h"

namespace {

// Lowers array.store by pulling hardware results into scratch before copy-back.
class ArrayStoreLowering
    : public mlir::OpConversionPattern<mlir::analog::ArrayStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Emits the runtime store call and writes its lane results into the dest tile.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::ArrayStoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Validate the store indices and lowered destination layout up front.
    if (adaptor.getIndices().size() < 2) {
      return rewriter.notifyMatchFailure(
          op, "expected at least [arrayRow, arrayCol] indices");
    }

    auto destType = llvm::dyn_cast<mlir::MemRefType>(adaptor.getDest().getType());
    if (!destType || destType.getRank() < 3) {
      return rewriter.notifyMatchFailure(
          op, "expected memref<gridR x gridC x lanes x elem>");
    }

    int64_t arrayRows = destType.getShape()[2];
    mlir::Value c0 = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);

    // Narrow the destination to the selected array result slice.
    llvm::SmallVector<mlir::OpFoldResult> offsets{
        adaptor.getIndices()[0], adaptor.getIndices()[1], c0};
    llvm::SmallVector<mlir::OpFoldResult> sizes{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(arrayRows)};
    llvm::SmallVector<mlir::OpFoldResult> strides{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};

    mlir::Value arrayMemref =
        rewriter
            .create<mlir::memref::SubViewOp>(op.getLoc(), adaptor.getDest(),
                                             offsets, sizes, strides)
            .getResult();

    // Allocate the ABI-shaped scratch buffer populated by the runtime call.
    auto scratchType =
        mlir::MemRefType::get({1, 1, arrayRows}, destType.getElementType());
    auto alignment = rewriter.getI64IntegerAttr(64);
    mlir::Value scratch =
        rewriter.create<mlir::memref::AllocOp>(op.getLoc(), scratchType,
                                               mlir::ValueRange{}, alignment);

    auto gridType =
        llvm::dyn_cast<mlir::analog::MatrixGridType>(op.getGrid().getType());
    if (!gridType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected analog.matrix.grid input type");
    }

    // Encode the grid coordinate and request the selected array result.
    int64_t gridCols = gridType.getGridShape()[1];
    mlir::Value arrayId = mlir::analog::golem::buildLinearArrayId(
        rewriter, op.getLoc(), adaptor.getIndices()[0], adaptor.getIndices()[1],
        gridCols);
    mlir::analog::golem::emitIntrinsicCall(
        rewriter, op.getLoc(), mlir::analog::golem::kStoreIntrinsicName,
        {scratch, arrayId});

    // Copy the runtime-populated lanes into the caller-visible destination.
    mlir::Value c1 = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 1);
    mlir::Value cArrayRows =
        rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), arrayRows);
    rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), c0, cArrayRows, c1, mlir::ValueRange{},
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value laneIndex,
            mlir::ValueRange) {
          mlir::Value value = builder.create<mlir::memref::LoadOp>(
              loc, scratch, mlir::ValueRange{c0, c0, laneIndex});
          builder.create<mlir::memref::StoreOp>(
              loc, value, arrayMemref, mlir::ValueRange{c0, c0, laneIndex});
          builder.create<mlir::scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace analog {
namespace golem {

// Registers store lowering for the Golem conversion pipeline.
void populateLowerStorePatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx) {
  patterns.add<ArrayStoreLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace analog
} // namespace mlir
