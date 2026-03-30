#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TENSOR_CONSTANT_UTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TENSOR_CONSTANT_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analog {
namespace detail {

FailureOr<SmallVector<float>> extractFloatElements(arith::ConstantOp op);

arith::ConstantOp createDenseF32ResourceConstant(OpBuilder &builder, Location loc,
                                                 RankedTensorType type,
                                                 StringRef resourceName,
                                                 ArrayRef<float> values);

std::string makeNumberedResourceName(StringRef prefix, unsigned &counter);

void eraseIfDead(arith::ConstantOp op);

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_TENSOR_CONSTANT_UTILS_H
