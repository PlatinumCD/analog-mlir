#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_EXTRACTORUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_EXTRACTORUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace analog {
namespace extractor_utils {
namespace detail {

// Recognizes linalg.generic wrappers whose body computes one op and yields it.
template <typename BodyOpTy>
inline bool genericYieldsSingleOpResult(Operation *op) {
  auto generic = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
  if (!generic)
    return false;

  Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return false;

  Block &block = region.front();
  if (block.empty())
    return false;

  auto it = block.begin();
  auto e = block.end();

  auto bodyOp = llvm::dyn_cast<BodyOpTy>(&*it++);
  if (!bodyOp || it == e)
    return false;

  auto yield = llvm::dyn_cast<linalg::YieldOp>(&*it++);
  if (!yield || it != e)
    return false;

  return yield.getNumOperands() == 1 &&
         yield.getOperand(0) == bodyOp.getResult();
}

} // namespace detail

// Checks the raw operand count while treating null operations as non-matches.
inline bool hasOperands(Operation *op, unsigned count) {
  return op && op->getNumOperands() == count;
}

// Checks DPS input count so destination operands are not counted as data inputs.
inline bool hasInputs(Operation *op, unsigned count) {
  auto dpsOp = llvm::dyn_cast_or_null<DestinationStyleOpInterface>(op);
  return dpsOp && dpsOp.getNumDpsInputs() == count;
}

// Returns the producer operation for an SSA value in a matcher graph.
inline Operation *defOp(Value value) { return value.getDefiningOp(); }

// Looks through an operand only when the operation and index are valid.
inline Operation *defOp(Operation *op, unsigned operandIndex) {
  if (!op || operandIndex >= op->getNumOperands())
    return nullptr;

  return op->getOperand(operandIndex).getDefiningOp();
}

// Tests whether an operand has a producer before matching its shape.
inline bool hasDefOp(Operation *op, unsigned operandIndex) {
  return defOp(op, operandIndex) != nullptr;
}

// Casts a value producer to the expected op type without caller null checks.
template <typename OpTy>
inline OpTy defOpAs(Value value) {
  return llvm::dyn_cast_or_null<OpTy>(defOp(value));
}

// Fetches and casts an operand producer in one bounds-checked matcher step.
template <typename OpTy>
inline OpTy defOpAs(Operation *op, unsigned operandIndex) {
  return llvm::dyn_cast_or_null<OpTy>(defOp(op, operandIndex));
}

// Tests whether an operand is produced by the expected operation type.
template <typename OpTy>
inline bool inputIs(Operation *op, unsigned operandIndex) {
  return static_cast<bool>(defOpAs<OpTy>(op, operandIndex));
}

// Matches two operands against an ordered pair of producer operation types.
template <typename FirstOpTy, typename SecondOpTy>
inline bool inputsAre(Operation *op, unsigned firstIndex = 0,
                      unsigned secondIndex = 1) {
  return inputIs<FirstOpTy>(op, firstIndex) &&
         inputIs<SecondOpTy>(op, secondIndex);
}

// Matches two operands against a pair of producer types in either order.
template <typename FirstOpTy, typename SecondOpTy>
inline bool inputsAreEither(Operation *op, unsigned firstIndex = 0,
                            unsigned secondIndex = 1) {
  return inputsAre<FirstOpTy, SecondOpTy>(op, firstIndex, secondIndex) ||
         inputsAre<SecondOpTy, FirstOpTy>(op, firstIndex, secondIndex);
}

// Recognizes elementwise addf regions used to encode bias additions.
inline bool isAddfGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<arith::AddFOp>(op);
}

// Recognizes elementwise tanh regions used at recurrent layer outputs.
inline bool isTanhGeneric(Operation *op) {
  return detail::genericYieldsSingleOpResult<math::TanhOp>(op);
}

// Provides the descriptive operand-count spelling for matcher call sites.
inline bool hasOperandCount(Operation *op, unsigned count) {
  return hasOperands(op, count);
}

// Provides the descriptive DPS-input-count spelling for matcher call sites.
inline bool hasInputOperandCount(Operation *op, unsigned count) {
  return hasInputs(op, count);
}

// Provides the explicit producer-query spelling for value matchers.
inline Operation *getDefiningOp(Value value) { return defOp(value); }

// Provides the explicit producer-query spelling for operand matchers.
inline Operation *getDefiningOp(Operation *op, unsigned operandIndex) {
  return defOp(op, operandIndex);
}

// Provides the explicit operand producer-presence spelling.
inline bool operandHasDefiningOp(Operation *op, unsigned operandIndex) {
  return hasDefOp(op, operandIndex);
}

// Provides the explicit typed producer-query spelling for value matchers.
template <typename OpTy>
inline OpTy getDefiningOpOfType(Value value) {
  return defOpAs<OpTy>(value);
}

// Provides the explicit typed producer-query spelling for operand matchers.
template <typename OpTy>
inline OpTy getDefiningOpOfType(Operation *op, unsigned operandIndex) {
  return defOpAs<OpTy>(op, operandIndex);
}

// Provides the explicit typed operand-producer predicate spelling.
template <typename OpTy>
inline bool operandDefiningOpIs(Operation *op, unsigned operandIndex) {
  return inputIs<OpTy>(op, operandIndex);
}

// Provides the explicit ordered producer-pair matcher spelling.
template <typename FirstOpTy, typename SecondOpTy>
inline bool operandDefiningOpsMatchOrdered(Operation *op,
                                           unsigned firstIndex = 0,
                                           unsigned secondIndex = 1) {
  return inputsAre<FirstOpTy, SecondOpTy>(op, firstIndex, secondIndex);
}

// Provides the explicit commutative producer-pair matcher spelling.
template <typename FirstOpTy, typename SecondOpTy>
inline bool operandDefiningOpsMatchEither(Operation *op,
                                          unsigned firstIndex = 0,
                                          unsigned secondIndex = 1) {
  return inputsAreEither<FirstOpTy, SecondOpTy>(op, firstIndex, secondIndex);
}

} // namespace extractor_utils
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_EXTRACTORUTILS_H
