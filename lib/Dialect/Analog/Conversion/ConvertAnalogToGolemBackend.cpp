#include "analog-mlir/Dialect/Analog/Conversion/ConvertAnalogToGolemBackend.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogBase.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
namespace analog {

namespace {

constexpr StringLiteral kSetIntrinsicName = "golem_analog_mvm_set";
constexpr StringLiteral kLoadIntrinsicName = "golem_analog_mvm_load";
constexpr StringLiteral kComputeIntrinsicName = "golem_analog_mvm_compute";
constexpr StringLiteral kStoreIntrinsicName = "golem_analog_mvm_store";

struct MatrixPlacementPlan {
  int64_t arrayRows;
  int64_t arrayCols;
  int64_t gridCols;
  Value c0;
  Value c1;
  Value cArrayRows;
  Value cArrayCols;
  Value rowOffset;
  Value colOffset;
  Value copyRows;
  Value copyCols;
};

struct VectorPlacementPlan {
  int64_t arrayCols;
  int64_t gridCols;
  Value c0;
  Value c1;
  Value cArrayCols;
  Value colOffset;
  Value copyCols;
};

class MatrixFromTensorLowering;
class VectorFromTensorLowering;
class MatrixPartitionLowering;
class VectorPartitionLowering;
class ArrayMatrixPlaceLowering;
class ArrayVectorPlaceLowering;
class ArrayExecuteLowering;
class ArrayStoreLowering;


// Creates or reuses a private function declaration for a backend intrinsic
// so rewritten ops can emit stable call sites.
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


// Normalizes index and integer values into i32 operands expected by the
// backend intrinsic ABI.
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


// Packs matrix placement metadata into the hardware/debug array id layout
// used by backend programming intrinsics.
static Value buildPackedArrayId(PatternRewriter &rewriter, Location loc, Value row,
                                Value col, int64_t gridCols, int64_t matrixWidth) {
  // The debug/hardware matrix-programming ABI packs the physical tile width
  // into the high 16 bits and the linear array index into the low 16 bits.
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


// Computes the flattened array id used by execution and vector-loading
// intrinsics for a row and column coordinate.
static Value buildLinearArrayId(PatternRewriter &rewriter, Location loc, Value row,
                                Value col, int64_t gridCols) {
  Value rowI32 = castToI32(rewriter, loc, row);
  Value colI32 = castToI32(rewriter, loc, col);
  Value cGridCols = rewriter.create<arith::ConstantIntOp>(loc, gridCols, 32);
  Value rowBase = rewriter.create<arith::MulIOp>(loc, rowI32, cGridCols);
  return rewriter.create<arith::AddIOp>(loc, rowBase, colI32);
}


// Emits a call to the named backend intrinsic, creating the declaration
// lazily with the operand-derived function type.
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


// Builds a `to_buffer` conversion for the lowered tensor value so the
// backend path can operate on memrefs directly.
static Value materializeTensorMemref(PatternRewriter &rewriter, Location loc,
                                     Value tensor) {
  auto tensorTy = llvm::cast<RankedTensorType>(tensor.getType());
  auto memrefTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
  return rewriter.create<bufferization::ToBufferOp>(loc, memrefTy, tensor);
}


// Builds a zero attribute for the given element type so scratch buffers
// can be padded before partial tile copies.
static FailureOr<TypedAttr> getZeroAttrForElementType(PatternRewriter &rewriter,
                                                      Operation *op,
                                                      Type elementType) {
  TypedAttr zeroAttr = rewriter.getZeroAttr(elementType);
  if (!zeroAttr) {
    return rewriter.notifyMatchFailure(op, "expected zero-initializable element type");
  }
  return zeroAttr;
}


// Writes a rectangular zero fill into a 2D memref using nested loops so
// partial tiles start from a known padded state.
static void zeroFill2DMemref(PatternRewriter &rewriter, Location loc,
                             Value memrefValue, Value rowUpper, Value colUpper,
                             Value zero, Value c0, Value c1) {
  rewriter.create<scf::ForOp>(
      loc, c0, rowUpper, c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value rowIdx, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, c0, colUpper, c1, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value colIdx, ValueRange) {
              colBuilder.create<memref::StoreOp>(
                  colLoc, zero, memrefValue, ValueRange{rowIdx, colIdx});
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}


// Copies the in-bounds portion of a matrix tile into the scratch buffer
// used for physical array placement.
static void copyMatrixTileIntoScratch(PatternRewriter &rewriter, Location loc,
                                      Value fullMemref, Value arrayMemref,
                                      Value rowOffset, Value colOffset,
                                      Value copyRows, Value copyCols, Value c0,
                                      Value c1) {
  rewriter.create<scf::ForOp>(
      loc, c0, copyRows, c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value rowIdx, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, c0, copyCols, c1, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value colIdx, ValueRange) {
              Value srcRow =
                  colBuilder.create<arith::AddIOp>(colLoc, rowOffset, rowIdx);
              Value srcCol =
                  colBuilder.create<arith::AddIOp>(colLoc, colOffset, colIdx);
              Value value = colBuilder.create<memref::LoadOp>(
                  colLoc, fullMemref, ValueRange{srcRow, srcCol});
              colBuilder.create<memref::StoreOp>(
                  colLoc, value, arrayMemref, ValueRange{rowIdx, colIdx});
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}


// Copies a vector slice into the single-row scratch tile expected by the
// backend load intrinsic.
static void copyVectorSliceIntoScratch(PatternRewriter &rewriter, Location loc,
                                       Value fullMemref, Value arrayMemref,
                                       Value colOffset, Value copyCols, Value c0,
                                       Value c1) {
  rewriter.create<scf::ForOp>(
      loc, c0, copyCols, c1, ValueRange{},
      [&](OpBuilder &builder, Location loopLoc, Value j, ValueRange) {
        Value srcCol = builder.create<arith::AddIOp>(loopLoc, colOffset, j);
        Value value = builder.create<memref::LoadOp>(loopLoc, fullMemref,
                                                     ValueRange{c0, srcCol});
        builder.create<memref::StoreOp>(loopLoc, value, arrayMemref,
                                        ValueRange{c0, j});
        builder.create<scf::YieldOp>(loopLoc);
      });
}


// Returns the bounded copy extent between the remaining source span and the
// fixed physical tile extent.
static Value buildClampedCopyUpperBound(PatternRewriter &rewriter, Location loc,
                                        Value sourceUpper, Value offset,
                                        Value tileUpper) {
  Value remaining = rewriter.create<arith::SubIOp>(loc, sourceUpper, offset);
  Value needsClamp = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, remaining, tileUpper);
  return rewriter.create<arith::SelectOp>(loc, needsClamp, remaining, tileUpper);
}


// Computes the constants and bounded copy region used to place one matrix
// tile into its physical array scratch buffer.
static MatrixPlacementPlan
buildMatrixPlacementPlan(PatternRewriter &rewriter, analog::ArrayMatrixPlaceOp op,
                         Value rowIndex, Value colIndex, Value fullMemref,
                         analog::MatrixGridType gridTy) {
  auto arrayShape = gridTy.getArrayShape();
  int64_t arrayRows = arrayShape[0];
  int64_t arrayCols = arrayShape[1];
  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value cArrayRows = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayRows);
  Value cArrayCols = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayCols);
  Value rowOffset = rewriter.create<arith::MulIOp>(op.getLoc(), rowIndex, cArrayRows);
  Value colOffset = rewriter.create<arith::MulIOp>(op.getLoc(), colIndex, cArrayCols);
  Value matrixRows = rewriter.create<memref::DimOp>(op.getLoc(), fullMemref, 0);
  Value matrixCols = rewriter.create<memref::DimOp>(op.getLoc(), fullMemref, 1);
  Value copyRows =
      buildClampedCopyUpperBound(rewriter, op.getLoc(), matrixRows, rowOffset, cArrayRows);
  Value copyCols =
      buildClampedCopyUpperBound(rewriter, op.getLoc(), matrixCols, colOffset, cArrayCols);

  return {
      arrayRows,  arrayCols, gridTy.getGridShape()[1], c0,       c1,
      cArrayRows, cArrayCols, rowOffset,             colOffset, copyRows,
      copyCols,
  };
}


// Computes the constants and bounded copy region used to stage one vector
// slice before issuing the backend load intrinsic.
static VectorPlacementPlan
buildVectorPlacementPlan(PatternRewriter &rewriter, analog::ArrayVectorPlaceOp op,
                         Value sliceIndex, Value fullMemref,
                         analog::VectorSliceType sliceTy) {
  int64_t arrayCols = sliceTy.getArrayShape()[1];
  Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
  Value cArrayCols = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayCols);
  Value colOffset = rewriter.create<arith::MulIOp>(op.getLoc(), sliceIndex, cArrayCols);
  Value vectorCols = rewriter.create<memref::DimOp>(op.getLoc(), fullMemref, 1);
  Value copyCols = buildClampedCopyUpperBound(rewriter, op.getLoc(), vectorCols,
                                              colOffset, cArrayCols);

  return {arrayCols, sliceTy.getGridShape()[1], c0, c1, cArrayCols, colOffset,
          copyCols};
}


// Creates and zero-fills a scratch tile buffer with the requested shape so
// placement lowerings can copy partial data into it.
static FailureOr<Value> allocateZeroedScratchTile(PatternRewriter &rewriter,
                                                  Operation *op,
                                                  ArrayRef<int64_t> shape,
                                                  Type elementType, Value rowUpper,
                                                  Value colUpper, Value c0,
                                                  Value c1) {
  auto scratchTy = MemRefType::get(shape, elementType);
  Value scratch = rewriter.create<memref::AllocOp>(op->getLoc(), scratchTy);

  auto maybeZeroAttr = getZeroAttrForElementType(rewriter, op, elementType);
  if (failed(maybeZeroAttr))
    return failure();

  Value zero = rewriter.create<arith::ConstantOp>(op->getLoc(), elementType,
                                                  *maybeZeroAttr);
  zeroFill2DMemref(rewriter, op->getLoc(), scratch, rowUpper, colUpper, zero, c0,
                   c1);
  return scratch;
}


// Installs all analog type conversions used by the backend lowering so the
// pass setup stays separate from pattern registration.
static void populateAnalogTypeConversions(TypeConverter &typeConverter) {
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
}


// Registers the concrete op lowerings that remove analog operations in favor
// of backend calls and standard dialect IR.
static void populateConversionPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx) {
  patterns.add<MatrixFromTensorLowering, VectorFromTensorLowering,
               MatrixPartitionLowering, VectorPartitionLowering,
               ArrayMatrixPlaceLowering, ArrayVectorPlaceLowering,
               ArrayExecuteLowering, ArrayStoreLowering>(typeConverter, ctx);
}


// Lowers matrix materialization away once the type has been converted to a
// plain ranked tensor.
class MatrixFromTensorLowering : public OpConversionPattern<analog::MatrixFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Replaces the analog wrapper op with its already-converted tensor input
  // after type lowering has made the wrapper redundant.
  LogicalResult
  matchAndRewrite(analog::MatrixFromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};


// Lowers vector materialization away once the converted tensor value can
// flow directly through the IR.
class VectorFromTensorLowering : public OpConversionPattern<analog::VectorFromTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Replaces the analog wrapper op with its converted tensor input and
  // leaves no backend-specific work behind.
  LogicalResult
  matchAndRewrite(analog::VectorFromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};


// Removes matrix partition wrappers after type conversion because backend
// lowering works on the lowered tensor value directly.
class MatrixPartitionLowering : public OpConversionPattern<analog::MatrixPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Forwards the converted matrix operand and drops the analog partition op
  // from the rewritten IR.
  LogicalResult
  matchAndRewrite(analog::MatrixPartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getMatrix());
    return success();
  }
};


// Removes vector partition wrappers once the backend conversion reaches the
// lowered tensor form.
class VectorPartitionLowering : public OpConversionPattern<analog::VectorPartitionOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Replaces the analog vector partition op with its converted tensor
  // operand to keep the lowered IR minimal.
  LogicalResult
  matchAndRewrite(analog::VectorPartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getVector());
    return success();
  }
};


// Lowers matrix placement into scratch-buffer preparation plus a backend
// intrinsic call that programs one physical array tile.
class ArrayMatrixPlaceLowering : public OpConversionPattern<analog::ArrayMatrixPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Builds a padded tile buffer, computes its packed array id, and emits
  // the backend matrix-programming call.
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

    // Materialize one zero-padded physical tile per placement op, then pass a
    // contiguous host buffer to the backend shim/intrinsic call.
    Value fullMemref = materializeTensorMemref(rewriter, op.getLoc(), adaptor.getInput());
    MatrixPlacementPlan plan = buildMatrixPlacementPlan(
        rewriter, op, adaptor.getRowIndex(), adaptor.getColIndex(), fullMemref,
        gridTy);
    FailureOr<Value> maybeScratch = allocateZeroedScratchTile(
        rewriter, op, {plan.arrayRows, plan.arrayCols}, matrixTy.getElementType(),
        plan.cArrayRows, plan.cArrayCols, plan.c0, plan.c1);
    if (failed(maybeScratch))
      return failure();
    Value arrayMemref = *maybeScratch;

    copyMatrixTileIntoScratch(rewriter, op.getLoc(), fullMemref, arrayMemref,
                              plan.rowOffset, plan.colOffset, plan.copyRows,
                              plan.copyCols, plan.c0, plan.c1);

    // The hardware matrix-programming path consumes a contiguous physical tile.
    // After padding, the host buffer stride is always the full array width.
    int64_t matrixWidth = plan.arrayCols;
    Value arrayId = buildPackedArrayId(rewriter, op.getLoc(), adaptor.getRowIndex(),
                                     adaptor.getColIndex(), plan.gridCols,
                                     matrixWidth);

    emitIntrinsicCall(rewriter, op.getLoc(), kSetIntrinsicName,
                      {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return success();
  }
};


// Lowers vector placement into a padded scratch tile plus the backend
// load intrinsic used to stage input vectors.
class ArrayVectorPlaceLowering : public OpConversionPattern<analog::ArrayVectorPlaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Materializes the vector slice scratch buffer, computes the array id,
  // and emits the backend load call.
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

    // Vectors use the same physical-tile contract, but only one logical row is
    // populated before the load intrinsic is issued.
    Value fullMemref = materializeTensorMemref(rewriter, op.getLoc(), adaptor.getInput());
    VectorPlacementPlan plan = buildVectorPlacementPlan(
        rewriter, op, adaptor.getSliceIndex(), fullMemref, sliceTy);
    FailureOr<Value> maybeScratch = allocateZeroedScratchTile(
        rewriter, op, {1, plan.arrayCols}, vectorTy.getElementType(), plan.c1,
        plan.cArrayCols, plan.c0, plan.c1);
    if (failed(maybeScratch))
      return failure();
    Value arrayMemref = *maybeScratch;

    copyVectorSliceIntoScratch(rewriter, op.getLoc(), fullMemref, arrayMemref,
                               plan.colOffset, plan.copyCols, plan.c0, plan.c1);

    Value row = plan.c0;
    Value col = adaptor.getSliceIndex();

    if (adaptor.getIndices().size() >= 2) {
      row = adaptor.getIndices()[0];
      col = adaptor.getIndices()[1];
    }

    Value arrayId = buildLinearArrayId(rewriter, op.getLoc(), row, col,
                                       plan.gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), kLoadIntrinsicName,
                      {arrayMemref, arrayId});

    rewriter.eraseOp(op);
    return success();
  }
};


// Lowers analog array execution to the compute intrinsic and leaves behind
// an empty tensor placeholder for later result population.
class ArrayExecuteLowering : public OpConversionPattern<analog::ArrayExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Issues the backend compute call for one array coordinate and replaces
  // the execute op with a lowered result tensor shell.
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

    emitIntrinsicCall(rewriter, op.getLoc(), kComputeIntrinsicName, {arrayId});

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


// Lowers array result stores into backend store intrinsics followed by a
// copy from the scratch buffer into the destination memref view.
class ArrayStoreLowering : public OpConversionPattern<analog::ArrayStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;


  // Creates the destination slice, issues the store intrinsic, and copies
  // the returned lane data into the final buffer.
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

    auto scratchTy = MemRefType::get({1, 1, arrayRows}, destTy.getElementType());
    auto alignment = rewriter.getI64IntegerAttr(64);
    Value scratch = rewriter.create<memref::AllocOp>(op.getLoc(), scratchTy,
                                                     ValueRange{}, alignment);

    auto gridTy = llvm::dyn_cast<analog::MatrixGridType>(op.getGrid().getType());
    if (!gridTy) {
      return rewriter.notifyMatchFailure(op, "expected analog.matrix.grid input type");
    }

    int64_t gridCols = gridTy.getGridShape()[1];
    Value arrayId = buildLinearArrayId(rewriter, op.getLoc(),
                                     adaptor.getIndices()[0],
                                     adaptor.getIndices()[1], gridCols);

    emitIntrinsicCall(rewriter, op.getLoc(), kStoreIntrinsicName,
                      {scratch, arrayId});

    Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value cArrayRows = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), arrayRows);
    rewriter.create<scf::ForOp>(
        op.getLoc(), c0, cArrayRows, c1, ValueRange{},
        [&](OpBuilder &b, Location loc, Value laneIdx, ValueRange) {
          Value value = b.create<memref::LoadOp>(loc, scratch, ValueRange{c0, c0, laneIdx});
          b.create<memref::StoreOp>(loc, value, arrayMemref, ValueRange{c0, c0, laneIdx});
          b.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace


// Returns the command-line pipeline name used to invoke this conversion
// pass from tooling and tests.
llvm::StringRef ConvertAnalogToGolemBackendPass::getArgument() const {
  return "convert-analog-to-golem-backend";
}


// Summarizes that this pass lowers analog dialect constructs into the
// golem backend intrinsic ABI.
llvm::StringRef ConvertAnalogToGolemBackendPass::getDescription() const {
  return "Convert analog dialect ops and types into golem backend instructions";
}


// Registers the dialects needed by the conversion patterns and the IR
// forms produced during lowering.
void ConvertAnalogToGolemBackendPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
}


// Sets up the type converter, installs all analog-to-backend rewrite
// patterns, and applies the partial conversion to the module.
void ConvertAnalogToGolemBackendPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp module = getOperation();

  TypeConverter typeConverter;
  populateAnalogTypeConversions(typeConverter);

  RewritePatternSet patterns(ctx);
  populateConversionPatterns(patterns, typeConverter, ctx);

  ConversionTarget target(*ctx);
  target.addIllegalDialect<analog::AnalogDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}


// Creates the conversion pass instance used by pass registration and
// pipeline construction.
std::unique_ptr<mlir::Pass> createConvertAnalogToGolemBackendPass() {
  return std::make_unique<ConvertAnalogToGolemBackendPass>();
}

} // namespace analog
} // namespace mlir
