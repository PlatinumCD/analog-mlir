#include "analog-mlir/Dialect/Analog/Conversion/golem/GolemUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include <cctype>

namespace mlir {
namespace analog {
namespace golem {

namespace {

// Produces a symbol-safe suffix while preserving readable alphanumeric runs.
std::string sanitizeSymbolSuffix(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value) {
    unsigned char uc = static_cast<unsigned char>(c);
    result.push_back(std::isalnum(uc) ? c : '_');
  }
  return result;
}

// Finds an unused module symbol by appending a numeric disambiguator as needed.
std::string makeUniqueGlobalName(ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef value) {
  std::string baseName = (prefix + sanitizeSymbolSuffix(value)).str();
  if (!module.lookupSymbol(baseName))
    return baseName;

  unsigned index = 0;
  std::string globalName = baseName + "_" + std::to_string(index);
  while (module.lookupSymbol(globalName)) {
    ++index;
    globalName = baseName + "_" + std::to_string(index);
  }

  return globalName;
}

// Builds index constants through the shared builder used by the lowering.
Value buildIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

// Converts a tile coordinate into the source offset for that tile dimension.
Value buildTileOffset(OpBuilder &builder, Location loc, Value tileIndex,
                      Value tileSize) {
  return builder.create<arith::MulIOp>(loc, tileIndex, tileSize);
}

} // namespace

// Reuses a runtime intrinsic declaration or inserts a private one in the module.
func::FuncOp getOrCreateIntrinsicDecl(ModuleOp module, llvm::StringRef name,
                                      FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;

  OpBuilder builder(&module.getBodyRegion());
  auto function = builder.create<func::FuncOp>(module.getLoc(), name, type);
  function.setPrivate();
  return function;
}

// Returns the opaque pointer type expected by the LLVM-facing runtime ABI.
Type getOpaquePointerType(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

// Deduplicates string constants so repeated runtime labels share one global.
LLVM::GlobalOp getOrCreateStringConstant(OpBuilder &builder, Location loc,
                                         ModuleOp module,
                                         llvm::StringRef prefix,
                                         llvm::StringRef value) {
  llvm::SmallString<64> nullTerminated(value);
  nullTerminated.push_back('\0');

  auto i8Type = IntegerType::get(builder.getContext(), 8);
  auto globalType =
      LLVM::LLVMArrayType::get(i8Type, nullTerminated.size_in_bytes());
  auto stringAttr = builder.getStringAttr(nullTerminated);

  // Prefer an existing constant with the same bytes and LLVM array type.
  for (auto global : module.getOps<LLVM::GlobalOp>()) {
    if (global.getGlobalType() == globalType && global.getConstant() &&
        global.getValueAttr() == stringAttr) {
      return global;
    }
  }

  // Insert globals at module scope so later lowering can reference them by name.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  std::string globalName = makeUniqueGlobalName(module, prefix, value);
  return builder.create<LLVM::GlobalOp>(loc, globalType,
                                        /*isConstant=*/true,
                                        LLVM::Linkage::Internal, globalName,
                                        stringAttr);
}

// Derives the enclosing module from the rewrite insertion point before loading.
Value getOrCreateGlobalStringPtr(PatternRewriter &rewriter, Location loc,
                                 llvm::StringRef prefix,
                                 llvm::StringRef value) {
  auto module = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  return getOrCreateGlobalStringPtr(static_cast<OpBuilder &>(rewriter), loc,
                                    module, prefix, value);
}

// Emits an address plus zero-offset GEP to produce an i8-compatible string ptr.
Value getOrCreateGlobalStringPtr(OpBuilder &builder, Location loc,
                                 ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef value) {
  auto global = getOrCreateStringConstant(builder, loc, module, prefix, value);
  auto globalPtr = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext(), global.getAddrSpace()),
      global.getSymNameAttr());
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getGlobalType(),
      globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});
}

// Adapts MLIR integer-like values to the fixed i32 runtime argument width.
Value castToI32(PatternRewriter &rewriter, Location loc, Value value) {
  Type i32Type = rewriter.getI32Type();
  Type valueType = value.getType();

  if (valueType.isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, i32Type, value);

  if (valueType.isInteger(32))
    return value;

  if (auto intType = llvm::dyn_cast<IntegerType>(valueType)) {
    if (intType.getWidth() < 32)
      return rewriter.create<arith::ExtUIOp>(loc, i32Type, value);
    return rewriter.create<arith::TruncIOp>(loc, i32Type, value);
  }

  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

// Flattens a row/column hardware grid coordinate into the runtime array id.
Value buildLinearArrayId(PatternRewriter &rewriter, Location loc, Value row,
                         Value col, int64_t gridCols) {
  Value rowI32 = castToI32(rewriter, loc, row);
  Value colI32 = castToI32(rewriter, loc, col);
  Value cGridCols = rewriter.create<arith::ConstantIntOp>(loc, gridCols, 32);
  Value rowBase = rewriter.create<arith::MulIOp>(loc, rowI32, cGridCols);
  return rewriter.create<arith::AddIOp>(loc, rowBase, colI32);
}

// Materializes a void runtime call with a declaration matching the operands.
void emitIntrinsicCall(PatternRewriter &rewriter, Location loc,
                       llvm::StringRef intrinsicName, ValueRange operands) {
  auto module = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();

  // Derive the declaration signature from the already-converted call operands.
  llvm::SmallVector<Type> argumentTypes;
  argumentTypes.reserve(operands.size());
  for (Value operand : operands)
    argumentTypes.push_back(operand.getType());

  auto functionType = rewriter.getFunctionType(argumentTypes, TypeRange{});
  auto callee = getOrCreateIntrinsicDecl(module, intrinsicName, functionType);
  rewriter.create<func::CallOp>(loc, callee.getName(), TypeRange{}, operands);
}

// Converts a ranked tensor value into a same-shaped memref for loop lowering.
Value materializeTensorMemref(PatternRewriter &rewriter, Location loc,
                              Value tensor) {
  auto tensorType = llvm::cast<RankedTensorType>(tensor.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return rewriter.create<bufferization::ToBufferOp>(loc, memrefType, tensor);
}

// Reports a match failure when an element type cannot provide a zero constant.
FailureOr<TypedAttr> getZeroAttrForElementType(PatternRewriter &rewriter,
                                               Operation *op,
                                               Type elementType) {
  TypedAttr zeroAttr = rewriter.getZeroAttr(elementType);
  if (!zeroAttr) {
    return rewriter.notifyMatchFailure(op,
                                       "expected zero-initializable element type");
  }
  return zeroAttr;
}

// Emits rectangular zero-fill loops for scratch regions that may be partial.
void zeroFill2DMemref(PatternRewriter &rewriter, Location loc,
                      Value memrefValue, Value rowUpper, Value colUpper,
                      Value zero, Value c0, Value c1) {
  rewriter.create<scf::ForOp>(
      loc, c0, rowUpper, c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value rowIndex, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, c0, colUpper, c1, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value colIndex,
                ValueRange) {
              colBuilder.create<memref::StoreOp>(
                  colLoc, zero, memrefValue, ValueRange{rowIndex, colIndex});
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}

// Copies the live portion of a matrix tile from the full source into scratch.
void copyMatrixTileIntoScratch(PatternRewriter &rewriter, Location loc,
                               Value fullMemref, Value arrayMemref,
                               Value rowOffset, Value colOffset, Value copyRows,
                               Value copyCols, Value c0, Value c1) {
  rewriter.create<scf::ForOp>(
      loc, c0, copyRows, c1, ValueRange{},
      [&](OpBuilder &rowBuilder, Location rowLoc, Value rowIndex, ValueRange) {
        rowBuilder.create<scf::ForOp>(
            rowLoc, c0, copyCols, c1, ValueRange{},
            [&](OpBuilder &colBuilder, Location colLoc, Value colIndex,
                ValueRange) {
              Value sourceRow =
                  colBuilder.create<arith::AddIOp>(colLoc, rowOffset, rowIndex);
              Value sourceCol =
                  colBuilder.create<arith::AddIOp>(colLoc, colOffset, colIndex);
              Value value = colBuilder.create<memref::LoadOp>(
                  colLoc, fullMemref, ValueRange{sourceRow, sourceCol});
              colBuilder.create<memref::StoreOp>(
                  colLoc, value, arrayMemref, ValueRange{rowIndex, colIndex});
              colBuilder.create<scf::YieldOp>(colLoc);
            });
        rowBuilder.create<scf::YieldOp>(rowLoc);
      });
}

// Copies the live portion of a one-row vector slice into scratch storage.
void copyVectorSliceIntoScratch(PatternRewriter &rewriter, Location loc,
                                Value fullMemref, Value arrayMemref,
                                Value colOffset, Value copyCols, Value c0,
                                Value c1) {
  rewriter.create<scf::ForOp>(
      loc, c0, copyCols, c1, ValueRange{},
      [&](OpBuilder &builder, Location loopLoc, Value columnIndex, ValueRange) {
        Value sourceCol =
            builder.create<arith::AddIOp>(loopLoc, colOffset, columnIndex);
        Value value = builder.create<memref::LoadOp>(
            loopLoc, fullMemref, ValueRange{c0, sourceCol});
        builder.create<memref::StoreOp>(loopLoc, value, arrayMemref,
                                        ValueRange{c0, columnIndex});
        builder.create<scf::YieldOp>(loopLoc);
      });
}

// Clamps the copy extent so edge tiles never read past the source shape.
Value buildClampedCopyUpperBound(PatternRewriter &rewriter, Location loc,
                                 Value sourceUpper, Value offset,
                                 Value tileUpper) {
  Value remaining = rewriter.create<arith::SubIOp>(loc, sourceUpper, offset);
  Value needsClamp = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, remaining, tileUpper);
  return rewriter.create<arith::SelectOp>(loc, needsClamp, remaining,
                                          tileUpper);
}

// Gathers the static tile shape and dynamic bounds for one matrix placement.
MatrixPlacementPlan buildMatrixPlacementPlan(
    PatternRewriter &rewriter, analog::ArrayMatrixPlaceOp op, Value rowIndex,
    Value colIndex, Value fullMemref, analog::MatrixGridType gridType) {
  Location loc = op.getLoc();
  auto arrayShape = gridType.getArrayShape();
  int64_t arrayRows = arrayShape[0];
  int64_t arrayCols = arrayShape[1];
  Value c0 = buildIndexConstant(rewriter, loc, 0);
  Value c1 = buildIndexConstant(rewriter, loc, 1);
  Value cArrayRows = buildIndexConstant(rewriter, loc, arrayRows);
  Value cArrayCols = buildIndexConstant(rewriter, loc, arrayCols);
  Value rowOffset = buildTileOffset(rewriter, loc, rowIndex, cArrayRows);
  Value colOffset = buildTileOffset(rewriter, loc, colIndex, cArrayCols);

  // Read source dimensions at runtime, then clamp copies for boundary tiles.
  Value matrixRows = rewriter.create<memref::DimOp>(loc, fullMemref, 0);
  Value matrixCols = rewriter.create<memref::DimOp>(loc, fullMemref, 1);
  Value copyRows =
      buildClampedCopyUpperBound(rewriter, loc, matrixRows, rowOffset, cArrayRows);
  Value copyCols =
      buildClampedCopyUpperBound(rewriter, loc, matrixCols, colOffset, cArrayCols);

  MatrixPlacementPlan plan{arrayRows,
                           arrayCols,
                           gridType.getGridShape()[1],
                           c0,
                           c1,
                           cArrayRows,
                           cArrayCols,
                           rowOffset,
                           colOffset,
                           copyRows,
                           copyCols};
  return plan;
}

// Gathers the static slice shape and dynamic bounds for one vector placement.
VectorPlacementPlan buildVectorPlacementPlan(
    PatternRewriter &rewriter, analog::ArrayVectorPlaceOp op, Value sliceIndex,
    Value fullMemref, analog::VectorSliceType sliceType) {
  Location loc = op.getLoc();
  int64_t arrayCols = sliceType.getArrayShape()[1];
  Value c0 = buildIndexConstant(rewriter, loc, 0);
  Value c1 = buildIndexConstant(rewriter, loc, 1);
  Value cArrayCols = buildIndexConstant(rewriter, loc, arrayCols);
  Value colOffset = buildTileOffset(rewriter, loc, sliceIndex, cArrayCols);

  // Read the vector width at runtime so the final slice can be partial.
  Value vectorCols = rewriter.create<memref::DimOp>(loc, fullMemref, 1);
  Value copyCols =
      buildClampedCopyUpperBound(rewriter, loc, vectorCols, colOffset, cArrayCols);

  VectorPlacementPlan plan{arrayCols,
                           sliceType.getGridShape()[1],
                           c0,
                           c1,
                           cArrayCols,
                           colOffset,
                           copyCols};
  return plan;
}

// Allocates a fixed-shape scratch tile and clears its active copied region.
FailureOr<Value> allocateZeroedScratchTile(PatternRewriter &rewriter,
                                           Operation *op,
                                           llvm::ArrayRef<int64_t> shape,
                                           Type elementType, Value rowUpper,
                                           Value colUpper, Value c0, Value c1) {
  auto scratchType = MemRefType::get(shape, elementType);
  Value scratch = rewriter.create<memref::AllocOp>(op->getLoc(), scratchType);

  // Validate zero materialization before emitting loops that depend on it.
  auto maybeZeroAttr = getZeroAttrForElementType(rewriter, op, elementType);
  if (failed(maybeZeroAttr))
    return failure();

  Value zero = rewriter.create<arith::ConstantOp>(op->getLoc(), elementType,
                                                  *maybeZeroAttr);
  zeroFill2DMemref(rewriter, op->getLoc(), scratch, rowUpper, colUpper, zero,
                   c0, c1);
  return scratch;
}

// Registers type rewrites from Analog wrappers to their tensor payload shapes.
void populateAnalogTypeConversions(TypeConverter &typeConverter) {
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

} // namespace golem
} // namespace analog
} // namespace mlir
