#include "analog-mlir/Dialect/Analog/Conversion/golem/GolemUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace {

// Lowers array.execute ops to compute calls while preserving grid SSA flow.
class ArrayExecuteLowering
    : public mlir::OpConversionPattern<mlir::analog::ArrayExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Emits the runtime compute side effect for one hardware grid coordinate.
  mlir::LogicalResult
  matchAndRewrite(mlir::analog::ArrayExecuteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // Validate the Analog type and index contract before using grid geometry.
    auto gridType =
        llvm::dyn_cast<mlir::analog::MatrixGridType>(op.getGrid().getType());
    if (!gridType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected analog.matrix.grid result type");
    }

    if (adaptor.getIndices().size() < 2) {
      return rewriter.notifyMatchFailure(op,
                                         "expected [arrayRow, arrayCol] indices");
    }

    // Encode the grid coordinate into the runtime's linear array identifier.
    int64_t gridCols = gridType.getGridShape()[1];
    mlir::Value arrayId = mlir::analog::golem::buildLinearArrayId(
        rewriter, op.getLoc(), adaptor.getIndices()[0], adaptor.getIndices()[1],
        gridCols);
    mlir::analog::golem::emitIntrinsicCall(
        rewriter, op.getLoc(),
        mlir::analog::golem::kComputeIntrinsicName, {arrayId});

    // Keep downstream grid users connected after emitting the compute call.
    auto loweredType = getTypeConverter()->convertType(op.getGrid().getType());
    auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(loweredType);
    if (!rankedType) {
      return rewriter.notifyMatchFailure(
          op, "expected analog.array.execute to lower to ranked tensor type");
    }

    mlir::Value placeholder = rewriter.create<mlir::tensor::EmptyOp>(
        op.getLoc(), rankedType.getShape(), rankedType.getElementType());
    rewriter.replaceOp(op, placeholder);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace analog {
namespace golem {

// Registers the array execution pattern for the Golem conversion pipeline.
void populateLowerExecutePatterns(RewritePatternSet &patterns,
                                  TypeConverter &typeConverter,
                                  MLIRContext *ctx) {
  patterns.add<ArrayExecuteLowering>(typeConverter, ctx);
}

} // namespace golem
} // namespace analog
} // namespace mlir
