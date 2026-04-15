#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_MATCHUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_MATCHUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace analog {
namespace match_utils {

// Tests pointer identity membership in an extractor's matched operation set.
inline bool containsOp(llvm::ArrayRef<Operation *> ops, Operation *op) {
  for (Operation *matchedOp : ops) {
    if (matchedOp == op)
      return true;
  }
  return false;
}

// Tests SSA value membership while preserving the caller's match ordering.
inline bool containsValue(llvm::ArrayRef<Value> values, Value value) {
  for (Value existing : values) {
    if (existing == value)
      return true;
  }
  return false;
}

// Records an optional matched operation once in the extraction order.
inline void appendUniqueOp(llvm::SmallVectorImpl<Operation *> &ops,
                           Operation *op) {
  if (!op || containsOp(ops, op))
    return;
  ops.push_back(op);
}

// Records a boundary value once so generated function signatures stay minimal.
inline void appendUniqueValue(llvm::SmallVectorImpl<Value> &values,
                              Value value) {
  if (containsValue(values, value))
    return;
  values.push_back(value);
}

// Builds the external input boundary by excluding values produced inside
// the matched subgraph.
inline void collectInputs(llvm::ArrayRef<Operation *> matchedOps,
                          llvm::SmallVectorImpl<Value> &inputs) {
  inputs.clear();

  for (Operation *op : matchedOps) {
    for (Value operand : op->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && containsOp(matchedOps, definingOp))
        continue;
      appendUniqueValue(inputs, operand);
    }
  }
}

// Builds the output boundary from the root operation replaced by the call.
inline void collectOutputs(Operation *root,
                           llvm::SmallVectorImpl<Value> &outputs) {
  outputs.clear();
  if (!root)
    return;

  for (Value result : root->getResults())
    appendUniqueValue(outputs, result);
}

} // namespace match_utils
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_MATCHUTILS_H
