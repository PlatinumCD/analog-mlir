#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_CONVERTERS_CONVERTERUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_CONVERTERS_CONVERTERUTILS_H

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace analog {
namespace converter_utils {

// Reads the extracted layer kind used to select a converter.
inline StringAttr getLayerType(Operation *op) {
  if (!op)
    return {};
  return op->getAttrOfType<StringAttr>("layer_type");
}

// Reads whether a layer function is still digital or already analog.
inline StringAttr getLayerDomain(Operation *op) {
  if (!op)
    return {};
  return op->getAttrOfType<StringAttr>("layer_domain");
}

// Compares a layer_type attribute while treating missing metadata as no match.
inline bool hasLayerType(Operation *op, StringRef layerType) {
  auto attr = getLayerType(op);
  return attr && attr.getValue() == layerType;
}

// Compares a layer_domain attribute while treating missing metadata as no match.
inline bool hasLayerDomain(Operation *op, StringRef layerDomain) {
  auto attr = getLayerDomain(op);
  return attr && attr.getValue() == layerDomain;
}

// Identifies extracted layer functions that still need analog conversion.
inline bool isDigitalLayer(Operation *op) {
  return getLayerType(op) && hasLayerDomain(op, "digital");
}

// Tracks the next stable ids that tie matrix and vector stages together.
extern int64_t nextMatrixId;
extern int64_t nextVectorId;

// Retains materialized values so later stages can recover id-tagged IR objects.
extern llvm::SmallVector<Value, 16> processedMatrices;
extern llvm::SmallVector<Value, 16> processedVectors;

// Attaches or reuses the matrix_id on a materialized matrix-producing op.
FailureOr<int64_t> getOrSetMatrixId(Value materializedMatrix,
                                    OpBuilder &builder);

// Attaches or reuses the vector_id on a materialized vector-producing op.
FailureOr<int64_t> getOrSetVectorId(Value materializedVector,
                                    OpBuilder &builder);

// Finds a previously materialized matrix by the id shared with its vector work.
FailureOr<Value> getMatrixWithId(int64_t matrixId);

// Wraps a 2D f32 weight constant in analog matrix IR and records its id.
FailureOr<Value> materializeAnalogMatrix(arith::ConstantOp weightConstant,
                                         OpBuilder &builder);

// Tiles a materialized matrix into an analog array grid of the requested shape.
FailureOr<Value> partitionAnalogMatrix(Value materializedMatrix,
                                       OpBuilder &builder,
                                       int64_t arrayRows,
                                       int64_t arrayCols);

// Emits placement loops that load each matrix tile onto its target array.
FailureOr<scf::ForOp> placeAnalogMatrix(Value partitionedMatrix, OpBuilder &builder);

// Wraps a 2D f32 tensor or block argument in analog vector IR and records it.
FailureOr<Value> materializeAnalogVector(Value value, OpBuilder &builder);

// Wraps a 2D f32 tensor in analog vector IR while explicitly pairing it with
// the matrix id that will consume it.
FailureOr<Value> materializeAnalogVector(Value value, int64_t vectorId,
                                         OpBuilder &builder);

// Slices a materialized vector to match the matrix grid used for execution.
FailureOr<Value> partitionAnalogVector(Value materializedVector,
                                       OpBuilder &builder,
                                       int64_t arrayRows,
                                       int64_t arrayCols);

// Emits placement loops that load each vector slice onto its target array.
FailureOr<scf::ForOp> placeAnalogVector(Value partitionedVector, OpBuilder &builder);

// Inserts array execution after placement and returns the per-tile output buffer.
FailureOr<Value> insertArrayExecution(Value placedMatrix, Value placedVector,
                                      scf::ForOp matrixPlacementLoop,
                                      scf::ForOp vectorPlacementLoop,
                                      OpBuilder &builder);

// Reduces array output lanes and replaces the original matmul result uses.
FailureOr<Value> insertArrayReduction(Value executionBuffer,
                                      Value partitionedMatrix,
                                      linalg::MatmulOp matmulOp,
                                      OpBuilder &builder);

} // namespace converter_utils
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_CONVERTERS_CONVERTERUTILS_H
