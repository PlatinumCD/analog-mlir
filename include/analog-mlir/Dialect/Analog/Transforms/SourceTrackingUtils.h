#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_SOURCE_TRACKING_UTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_SOURCE_TRACKING_UTILS_H

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace analog {
namespace detail {

llvm::DenseMap<int64_t, analog::MatrixGridType>
collectMatrixGridsBySourceId(func::FuncOp func);

IntegerAttr getMatrixSourceIdAttr(Operation *op);

IntegerAttr findMatrixSourceIdOnValue(Value value);

IntegerAttr getOrInferMatmulSourceId(linalg::MatmulOp op);

int64_t getNextMatrixSourceId(func::FuncOp func);

IntegerAttr getOrCreateMatrixSourceId(arith::ConstantOp op,
                                      int64_t &nextMatrixSourceId);

void propagateMatrixSourceId(arith::ConstantOp op, Operation *materializedOp);

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_SOURCE_TRACKING_UTILS_H
