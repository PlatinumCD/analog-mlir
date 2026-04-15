#ifndef ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_GOLEM_GOLEMUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_GOLEM_GOLEMUTILS_H

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace analog {
namespace golem {

// Runtime intrinsic names emitted by the Golem backend lowering patterns.
inline constexpr llvm::StringLiteral kSetIntrinsicName = "golem_analog_mvm_set";
inline constexpr llvm::StringLiteral kLoadIntrinsicName = "golem_analog_mvm_load";
inline constexpr llvm::StringLiteral kComputeIntrinsicName =
    "golem_analog_mvm_compute";
inline constexpr llvm::StringLiteral kStoreIntrinsicName =
    "golem_analog_mvm_store";

// Carries static tile shape and dynamic bounds for placing one matrix tile.
struct MatrixPlacementPlan {
  // Static hardware shape used for scratch allocation and array id math.
  int64_t arrayRows;
  int64_t arrayCols;
  int64_t gridCols;

  // Reusable index constants keep generated loop bounds consistent.
  Value c0;
  Value c1;
  Value cArrayRows;
  Value cArrayCols;

  // Source offsets and clamped extents describe the active edge tile region.
  Value rowOffset;
  Value colOffset;
  Value copyRows;
  Value copyCols;
};

// Carries static slice shape and dynamic bounds for placing one vector slice.
struct VectorPlacementPlan {
  // Static hardware shape used for scratch allocation and array id math.
  int64_t arrayCols;
  int64_t gridCols;

  // Reusable index constants keep generated loop bounds consistent.
  Value c0;
  Value c1;
  Value cArrayCols;

  // Source offset and clamped extent describe the active edge slice region.
  Value colOffset;
  Value copyCols;
};

// Returns a private intrinsic declaration, creating it in the module if needed.
func::FuncOp getOrCreateIntrinsicDecl(ModuleOp module, llvm::StringRef name,
                                      FunctionType type);

// Centralizes the opaque LLVM pointer type used by runtime ABI values.
Type getOpaquePointerType(MLIRContext *context);

// Deduplicates an LLVM string global by contents before creating a new constant.
LLVM::GlobalOp getOrCreateStringConstant(OpBuilder &builder, Location loc,
                                         ModuleOp module, llvm::StringRef prefix,
                                         llvm::StringRef value);

// Emits an LLVM pointer to a deduplicated global string in the given module.
Value getOrCreateGlobalStringPtr(OpBuilder &builder, Location loc,
                                 ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef value);

// Finds the enclosing module from the rewrite point before emitting the pointer.
Value getOrCreateGlobalStringPtr(PatternRewriter &rewriter, Location loc,
                                 llvm::StringRef prefix,
                                 llvm::StringRef value);

// Normalizes values to the i32 Golem ABI, using zero for unsupported types.
Value castToI32(PatternRewriter &rewriter, Location loc, Value value);

// Computes the row-major hardware array id for grid coordinates.
Value buildLinearArrayId(PatternRewriter &rewriter, Location loc, Value row,
                         Value col, int64_t gridCols);

// Declares the intrinsic if needed and emits a void call with operand types.
void emitIntrinsicCall(PatternRewriter &rewriter, Location loc,
                       llvm::StringRef intrinsicName, ValueRange operands);

// Buffers an already-ranked tensor as a same-shaped memref for scratch copies.
Value materializeTensorMemref(PatternRewriter &rewriter, Location loc,
                              Value tensor);

// Returns a zero attribute or reports a match failure for unsupported elements.
FailureOr<TypedAttr> getZeroAttrForElementType(PatternRewriter &rewriter,
                                               Operation *op,
                                               Type elementType);

// Emits nested loops that zero-initialize the active rectangle of a 2D memref.
void zeroFill2DMemref(PatternRewriter &rewriter, Location loc,
                      Value memrefValue, Value rowUpper, Value colUpper,
                      Value zero, Value c0, Value c1);

// Copies a clamped matrix tile from the full buffer into scratch memory.
void copyMatrixTileIntoScratch(PatternRewriter &rewriter, Location loc,
                               Value fullMemref, Value arrayMemref,
                               Value rowOffset, Value colOffset, Value copyRows,
                               Value copyCols, Value c0, Value c1);

// Copies a clamped vector slice from the full buffer into scratch memory.
void copyVectorSliceIntoScratch(PatternRewriter &rewriter, Location loc,
                                Value fullMemref, Value arrayMemref,
                                Value colOffset, Value copyCols, Value c0,
                                Value c1);

// Computes min(sourceUpper - offset, tileUpper) for partial edge tiles.
Value buildClampedCopyUpperBound(PatternRewriter &rewriter, Location loc,
                                 Value sourceUpper, Value offset,
                                 Value tileUpper);

// Builds the constants, offsets, and copy bounds shared by matrix placement.
MatrixPlacementPlan buildMatrixPlacementPlan(
    PatternRewriter &rewriter, analog::ArrayMatrixPlaceOp op, Value rowIndex,
    Value colIndex, Value fullMemref, analog::MatrixGridType gridTy);

// Builds the constants, offsets, and copy bounds shared by vector placement.
VectorPlacementPlan buildVectorPlacementPlan(
    PatternRewriter &rewriter, analog::ArrayVectorPlaceOp op, Value sliceIndex,
    Value fullMemref, analog::VectorSliceType sliceTy);

// Allocates scratch memory and fails if the element type cannot be zero-filled.
FailureOr<Value> allocateZeroedScratchTile(PatternRewriter &rewriter,
                                           Operation *op,
                                           llvm::ArrayRef<int64_t> shape,
                                           Type elementType, Value rowUpper,
                                           Value colUpper, Value c0, Value c1);

// Maps Analog aggregate types to ranked tensors while preserving other types.
void populateAnalogTypeConversions(TypeConverter &typeConverter);

// Adds matrix lowering patterns that erase wrappers and emit Golem set calls.
void populateLowerMatrixPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx);

// Adds vector lowering patterns that erase wrappers and emit Golem load calls.
void populateLowerVectorPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter,
                                 MLIRContext *ctx);

// Adds execute lowering patterns that emit Golem compute calls.
void populateLowerExecutePatterns(RewritePatternSet &patterns,
                                  TypeConverter &typeConverter,
                                  MLIRContext *ctx);

// Adds store lowering patterns that emit Golem store calls into destinations.
void populateLowerStorePatterns(RewritePatternSet &patterns,
                                TypeConverter &typeConverter,
                                MLIRContext *ctx);

} // namespace golem
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_CONVERSION_GOLEM_GOLEMUTILS_H
