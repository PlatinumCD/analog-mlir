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

static Value buildPackedArrayId(PatternRewriter &rewriter, Location loc, Value row,
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

static Value buildLinearArrayId(PatternRewriter &rewriter, Location loc, Value row,
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


class MatrixPartitionLowering : public OpConversionPattern<analog::MatrixPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::MatrixPartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getMatrix());
    return success();
  }
};


class VectorPartitionLowering : public OpConversionPattern<analog::VectorPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::VectorPartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getVector());
    return success();
  }
};


class ArrayMatrixPlaceLowering : public OpConversionPattern<analog::ArrayMatrixPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::ArrayMatrixPlaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto gridTy = llvm::dyn_cast<analog::MatrixGridType>(op.getInput().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.matrix.grid input type");
    }

    auto matrixTy = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!matrixTy || matrixTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected lowered matrix tensor<mxn>");
    }

    auto arrayShape = gridTy.getArrayShape();
    int64_t arrayRows = arrayShape[0];
    int64_t arrayCols = arrayShape[1];
    Value cArrayRows = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayRows);
    Value cArrayCols = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayCols);
    Value rowOffset = rewriter.create<arith::MulIOp>(op.getLoc(), adaptor.getRowIndex(), cArrayRows);
    Value colOffset = rewriter.create<arith::MulIOp>(op.getLoc(), adaptor.getColIndex(), cArrayCols);

    SmallVector<OpFoldResult> offsets{rowOffset, colOffset};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(arrayRows), rewriter.getIndexAttr(arrayCols)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

    auto fullMemrefTy = MemRefType::get(matrixTy.getShape(), matrixTy.getElementType());
    Value fullMemref =
        rewriter.create<bufferization::ToBufferOp>(op.getLoc(), fullMemrefTy, adaptor.getInput());
    auto subviewTy =
        memref::SubViewOp::inferResultType(fullMemrefTy, offsets, sizes, strides);
    Value arrayMemref =
        rewriter.create<memref::SubViewOp>(op.getLoc(), subviewTy, fullMemref, offsets, sizes, strides);

    int64_t gridCols = gridTy.getGridShape()[1];
    int64_t matrixWidth = matrixTy.getShape()[1];
    if (ShapedType::isDynamic(matrixWidth))
      matrixWidth = gridTy.getMatrix().getShape()[1];
    if (ShapedType::isDynamic(matrixWidth))
      return rewriter.notifyMatchFailure(op, "expected static matrix width for packed array_id");
    Value arrayId = buildPackedArrayId(rewriter, op.getLoc(), adaptor.getRowIndex(),
                                     adaptor.getColIndex(), gridCols,
                                     matrixWidth);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_set", {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return success();
  }
};

class ArrayVectorPlaceLowering : public OpConversionPattern<analog::ArrayVectorPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::ArrayVectorPlaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto sliceTy = llvm::dyn_cast<analog::VectorSliceType>(op.getInput().getType());
    if (!sliceTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.vector.slice input type");
    }

    auto vectorTy = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!vectorTy || vectorTy.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected lowered vector tensor<1xn>");
    }

    auto arrayShape = sliceTy.getArrayShape();
    int64_t arrayCols = arrayShape[1];
    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value cArrayCols = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayCols);
    Value colOffset = rewriter.create<arith::MulIOp>(op.getLoc(), adaptor.getSliceIndex(), cArrayCols);

    SmallVector<OpFoldResult> offsets{c0, colOffset};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(1), rewriter.getIndexAttr(arrayCols)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};

    auto fullMemrefTy = MemRefType::get(vectorTy.getShape(), vectorTy.getElementType());
    Value fullMemref = rewriter.create<bufferization::ToBufferOp>(op.getLoc(), fullMemrefTy, adaptor.getInput());
    auto subviewTy = memref::SubViewOp::inferResultType(fullMemrefTy, offsets, sizes, strides);
    Value arrayMemref = rewriter.create<memref::SubViewOp>(op.getLoc(), subviewTy, fullMemref, offsets, sizes, strides);
    int64_t gridCols = sliceTy.getGridShape()[1];
    Value row = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value col = adaptor.getSliceIndex();

    if (adaptor.getIndices().size() >= 2) {
      row = adaptor.getIndices()[0];
      col = adaptor.getIndices()[1];
    }

    Value arrayId = buildLinearArrayId(rewriter, op.getLoc(), row, col, gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_load", {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return success();
  }
};

class ArrayExecuteLowering : public OpConversionPattern<analog::ArrayExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::ArrayExecuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto gridTy = llvm::dyn_cast<analog::MatrixGridType>(op.getGrid().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.matrix.grid result type");
    }

    if (adaptor.getIndices().size() < 2) {
      return rewriter.notifyMatchFailure(op, "expected [arrayRow, arrayCol] indices");
    }

    int64_t gridCols = gridTy.getGridShape()[1];
    Value arrayId = buildLinearArrayId(rewriter, op.getLoc(),
                                     adaptor.getIndices()[0],
                                     adaptor.getIndices()[1], gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_compute", {arrayId});

    auto loweredTy = getTypeConverter()->convertType(op.getType());
    auto rankedTy = llvm::dyn_cast<RankedTensorType>(loweredTy);
    if (!rankedTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.array.execute to lower to ranked tensor type");
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

class ArrayStoreLowering : public OpConversionPattern<analog::ArrayStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(analog::ArrayStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (adaptor.getIndices().size() < 2) {
      return rewriter.notifyMatchFailure(op, "expected at least [arrayRow, arrayCol] indices");
    }

    auto destTy = llvm::dyn_cast<MemRefType>(adaptor.getDest().getType());
    if (!destTy || destTy.getRank() < 3) {
      return rewriter.notifyMatchFailure(op, "expected memref<gridR x gridC x lanes x elem>");
    }

    int64_t arrayRows = destTy.getShape()[2];
    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    SmallVector<OpFoldResult> offsets{adaptor.getIndices()[0], adaptor.getIndices()[1], c0};
    SmallVector<OpFoldResult> sizes{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(arrayRows)};
    SmallVector<OpFoldResult> strides{
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)};

    Value arrayMemref = rewriter
                           .create<memref::SubViewOp>(op.getLoc(), adaptor.getDest(), offsets, sizes, strides)
                           .getResult();

    auto gridTy = llvm::dyn_cast<analog::MatrixGridType>(op.getGrid().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.matrix.grid input type");
    }

    int64_t gridCols = gridTy.getGridShape()[1];
    Value arrayId = buildLinearArrayId(rewriter, op.getLoc(),
                                     adaptor.getIndices()[0],
                                     adaptor.getIndices()[1], gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), "golem_analog_mvm_store", {arrayMemref, arrayId});

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
  typeConverter.addConversion([](analog::MatrixGridType type) -> Type {
    auto matrix = type.getMatrix();
    return RankedTensorType::get(matrix.getShape(), matrix.getElementType());
  });
  typeConverter.addConversion([](analog::VectorSliceType type) -> Type {
    auto vector = type.getVector();
    return RankedTensorType::get(vector.getShape(), vector.getElementType());
  });

  RewritePatternSet patterns(ctx);
  patterns.add<MatrixFromTensorLowering, VectorFromTensorLowering,
               MatrixPartitionLowering, VectorPartitionLowering,
               ArrayMatrixPlaceLowering, ArrayVectorPlaceLowering,
               ArrayExecuteLowering, ArrayStoreLowering>(typeConverter, ctx);

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
