#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OUTLINING_UTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OUTLINING_UTILS_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace analog {
namespace detail {

std::string buildOutlinedFunctionName(StringRef prefix, int64_t id);

void cloneOutlinedOpsIntoBuilder(Block &source, OpBuilder &builder,
                                 IRMapping &mapper);

FailureOr<SmallVector<Value>> getOutlinedLayerReturns(Block &bodyBlk,
                                                      Block &exitBlk,
                                                      IRMapping &mapper);

} // namespace detail
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_OUTLINING_UTILS_H
