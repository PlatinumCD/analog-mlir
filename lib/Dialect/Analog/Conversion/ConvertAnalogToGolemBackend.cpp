#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

static func::FuncOp getOrCreateIntrinsicDecl(ModuleOp module, StringRef name,
                                             FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
    return existing;
  }

  OpBuilder b(module.getBodyRegion());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static Value castToI32(PatternRewriter &rewriter, Location loc, Value value) {
  Type i32Ty = rewriter.getI32Type();
  Type valueTy = value.getType();

  if (valueTy.isIndex()) {
    return rewriter.create<arith::IndexCastOp>(loc, i32Ty, value);
  }

  if (valueTy.isInteger(32)) {
    return value;
  }

  if (auto intTy = llvm::dyn_cast<IntegerType>(valueTy)) {
    if (intTy.getWidth() < 32) {
      return rewriter.create<arith::ExtUIOp>(loc, i32Ty, value);
    }
    return rewriter.create<arith::TruncIOp>(loc, i32Ty, value);
  }

  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

static Value buildPackedTileId(PatternRewriter &rewriter, Location loc, Value row,
                               Value col, int64_t gridCols, int64_t matrixWidth) {
  Value rowI32 = castToI32(rewriter, loc, row);
  Value colI32 = castToI32(rewriter, loc, col);
  Value cGridCols = rewriter.create<arith::ConstantIntOp>(loc, gridCols, 32);
  Value rowBase = rewriter.create<arith::MulIOp>(loc, rowI32, cGridCols);
  Value linearId = rewriter.create<arith::AddIOp>(loc, rowBase, colI32);

  Value cMask = rewriter.create<arith::ConstantIntOp>(loc, 0xFFFF, 32);
  Value cMatrixWidth = rewriter.create<arith::ConstantIntOp>(loc, matrixWidth, 32);
  Value width16 = rewriter.create<arith::AndIOp>(loc, cMatrixWidth, cMask);
  Value cShift = rewriter.create<arith::ConstantIntOp>(loc, 16, 32);
  Value upper = rewriter.create<arith::ShLIOp>(loc, width16, cShift);
  Value lower = rewriter.create<arith::AndIOp>(loc, linearId, cMask);
  return rewriter.create<arith::OrIOp>(loc, upper, lower);
}

static Value buildLinearTileId(PatternRewriter &rewriter, Location loc, Value row,
                               Value col, int64_t gridCols) {
  Value rowI32 = castToI32(rewriter, loc, row);
  Value colI32 = castToI32(rewriter, loc, col);
  Value cGridCols = rewriter.create<arith::ConstantIntOp>(loc, gridCols, 32);
  Value rowBase = rewriter.create<arith::MulIOp>(loc, rowI32, cGridCols);
  return rewriter.create<arith::AddIOp>(loc, rowBase, colI32);
}

static void emitIntrinsicCall(PatternRewriter &rewriter, Location loc,
                              StringRef intrinsicName, ValueRange operands) {

  auto module = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  SmallVector<Type> argTypes;
  argTypes.reserve(operands.size());
  for (Value v : operands) {
    argTypes.push_back(v.getType());
  }

  auto fnType = rewriter.getFunctionType(argTypes, TypeRange{});
  auto callee = getOrCreateIntrinsicDecl(module, intrinsicName, fnType);
  rewriter.create<func::CallOp>(loc, callee.getName(), TypeRange{}, operands);
}


class MatrixFromTensorLowering : public OpConversionPattern<analog::MatrixFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::MatrixFromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};


class VectorFromTensorLowering : public OpConversionPattern<analog::VectorFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::VectorFromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};


class TilePartitionLowering : public OpConversionPattern<analog::TilePartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::TilePartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getMatrix());
    return success();
  }
};


class VTilePartitionLowering : public OpConversionPattern<analog::VTilePartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::VTilePartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getVector());
    return success();
  }
};


class TilePlaceLowering : public OpConversionPattern<analog::TilePlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::TilePlaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto gridTy = llvm::dyn_cast<analog::TileGridType>(op.getInput().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.tile.grid input type");
    }

    auto matrixTy = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!matrixTy || matrixTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected lowered matrix tensor<mxn>");
    }

    auto tileShape = gridTy.getTileShape();
    int64_t tileRows = tileShape[0];
    int64_t tileCols = tileShape[1];
    Value cTileRows = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), tileRows);
    Value cTileCols = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), tileCols);
    Value rowOffset = rewriter.create<arith::MulIOp>(op.getLoc(), adaptor.getRowIndex(), cTileRows);
    Value colOffset = rewriter.create<arith::MulIOp>(op.getLoc(), adaptor.getColIndex(), cTileCols);

    SmallVector<OpFoldResult> offsets{rowOffset, colOffset};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(tileRows), rewriter.getIndexAttr(tileCols)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

    auto fullMemrefTy = MemRefType::get(matrixTy.getShape(), matrixTy.getElementType());
    Value fullMemref =
        rewriter.create<bufferization::ToBufferOp>(op.getLoc(), fullMemrefTy, adaptor.getInput());
    auto subviewTy =
        memref::SubViewOp::inferResultType(fullMemrefTy, offsets, sizes, strides);
    Value tileMemref =
        rewriter.create<memref::SubViewOp>(op.getLoc(), subviewTy, fullMemref, offsets, sizes, strides);

    int64_t gridCols = gridTy.getGridShape()[1];
    int64_t matrixWidth = matrixTy.getShape()[1];
    if (ShapedType::isDynamic(matrixWidth))
      matrixWidth = gridTy.getMatrix().getShape()[1];
    if (ShapedType::isDynamic(matrixWidth))
      return rewriter.notifyMatchFailure(op, "expected static matrix width for packed tile_id");
    Value tileId = buildPackedTileId(rewriter, op.getLoc(), adaptor.getRowIndex(),
                                     adaptor.getColIndex(), gridCols,
                                     matrixWidth);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_set", {tileMemref, tileId});

    rewriter.eraseOp(op);
    return success();
  }
};

class VTilePlaceLowering : public OpConversionPattern<analog::VTilePlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::VTilePlaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto sliceTy = llvm::dyn_cast<analog::VTileSliceType>(op.getInput().getType());
    if (!sliceTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.vtile.slice input type");
    }

    auto vectorTy = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!vectorTy || vectorTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected lowered vector tensor<1xn>");
    }

    auto tileShape = sliceTy.getTileShape();
    int64_t tileCols = tileShape[1];
    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value cTileCols = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), tileCols);
    Value colOffset = rewriter.create<arith::MulIOp>(op.getLoc(), adaptor.getSliceIndex(), cTileCols);

    SmallVector<OpFoldResult> offsets{c0, colOffset};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1), rewriter.getIndexAttr(tileCols)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

    auto fullMemrefTy = MemRefType::get(vectorTy.getShape(), vectorTy.getElementType());
    Value fullMemref = rewriter.create<bufferization::ToBufferOp>(op.getLoc(), fullMemrefTy, adaptor.getInput());
    auto subviewTy = memref::SubViewOp::inferResultType(fullMemrefTy, offsets, sizes, strides);
    Value tileMemref = rewriter.create<memref::SubViewOp>(op.getLoc(), subviewTy, fullMemref, offsets, sizes, strides);
    int64_t gridCols = sliceTy.getGridShape()[1];
    Value row = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value col = adaptor.getSliceIndex();

    if (adaptor.getIndices().size() >= 2) {
      row = adaptor.getIndices()[0];
      col = adaptor.getIndices()[1];
    }

    Value tileId = buildLinearTileId(rewriter, op.getLoc(), row, col, gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_load", {tileMemref, tileId});

    rewriter.eraseOp(op);
    return success();
  }
};

class TileExecuteLowering : public OpConversionPattern<analog::TileExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::TileExecuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto gridTy = llvm::dyn_cast<analog::TileGridType>(op.getGrid().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.tile.grid result type");
    }

    if (adaptor.getIndices().size() < 2) {
      return rewriter.notifyMatchFailure(op, "expected [tileRow, tileCol] indices");
    }

    int64_t gridCols = gridTy.getGridShape()[1];
    Value tileId = buildLinearTileId(rewriter, op.getLoc(),
                                     adaptor.getIndices()[0],
                                     adaptor.getIndices()[1], gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_compute", {tileId});

    auto loweredTy = getTypeConverter()->convertType(op.getType());
    auto rankedTy = llvm::dyn_cast<RankedTensorType>(loweredTy);
    if (!rankedTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.tile.execute to lower to ranked tensor type");
    }

    Value lowered = rewriter.create<tensor::EmptyOp>(
        op.getLoc(),
        rankedTy.getShape(),
        rankedTy.getElementType()
    );

    rewriter.replaceOp(op, lowered);
    return success();
  }
};

class TileStoreLowering : public OpConversionPattern<analog::TileStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::TileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (adaptor.getIndices().size() < 2) {
      return rewriter.notifyMatchFailure(op, "expected at least [tileRow, tileCol] indices");
    }

    auto destTy = llvm::dyn_cast<MemRefType>(adaptor.getDest().getType());
    if (!destTy || destTy.getRank() < 3) {
      return rewriter.notifyMatchFailure(op, "expected memref<gridR x gridC x lanes x elem>");
    }

    int64_t tileRows = destTy.getShape()[2];
    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    SmallVector<OpFoldResult> offsets{adaptor.getIndices()[0], adaptor.getIndices()[1], c0};
    SmallVector<OpFoldResult> sizes{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(tileRows)};
    SmallVector<OpFoldResult> strides{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};

    Value tileMemref = rewriter
                           .create<memref::SubViewOp>(op.getLoc(), adaptor.getDest(), offsets, sizes, strides)
                           .getResult();

    auto gridTy = llvm::dyn_cast<analog::TileGridType>(op.getGrid().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.tile.grid input type");
    }

    int64_t gridCols = gridTy.getGridShape()[1];
    Value tileId = buildLinearTileId(rewriter, op.getLoc(),
                                     adaptor.getIndices()[0],
                                     adaptor.getIndices()[1], gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_store", {tileMemref, tileId});

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

llvm::StringRef ConvertAnalogToGolemBackendPass::getArgument() const {
  return "convert-analog-to-golem-backend";
}

llvm::StringRef ConvertAnalogToGolemBackendPass::getDescription() const {
  return "Convert analog dialect ops and types into golem backend instructions";
}

void ConvertAnalogToGolemBackendPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
}

void ConvertAnalogToGolemBackendPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  mlir::func::FuncOp func = getOperation();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([](analog::MatrixType type) -> Type {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addConversion([](analog::VectorType type) -> Type {
    return RankedTensorType::get(type.getShape(), type.getElementType());
  });
  typeConverter.addConversion([](analog::TileGridType type) -> Type {
    auto matrix = type.getMatrix();
    return RankedTensorType::get(matrix.getShape(), matrix.getElementType());
  });
  typeConverter.addConversion([](analog::VTileSliceType type) -> Type {
    auto vector = type.getVector();
    return RankedTensorType::get(vector.getShape(), vector.getElementType());
  });

  RewritePatternSet patterns(ctx);
  patterns.add<MatrixFromTensorLowering, VectorFromTensorLowering,
               TilePartitionLowering, VTilePartitionLowering,
               TilePlaceLowering, VTilePlaceLowering, TileExecuteLowering,
               TileStoreLowering>(typeConverter, ctx);

  ConversionTarget target(*ctx);
  target.addIllegalDialect<analog::AnalogDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createConvertAnalogToGolemBackendPass() {
  return std::make_unique<ConvertAnalogToGolemBackendPass>();
}

} // namespace analog
} // namespace mlir
