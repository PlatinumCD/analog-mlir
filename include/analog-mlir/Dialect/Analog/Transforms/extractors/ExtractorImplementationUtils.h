#ifndef ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_EXTRACTORIMPLEMENTATIONUTILS_H
#define ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_EXTRACTORIMPLEMENTATIONUTILS_H

#include "analog-mlir/Dialect/Analog/Transforms/extractors/ExtractorUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"

#include <optional>

namespace mlir {
namespace analog {
namespace extractor_impl {

// Carries the bias, empty tensor, and broadcast op that seed a biased output.
struct BiasOutputInit {
  Operation *biasConstant = nullptr;
  Operation *outputEmpty = nullptr;
  Operation *outputBroadcast = nullptr;
};

// Carries the empty tensor, fill op, and scalar constant for bias-free outputs.
struct FillOutputInit {
  Operation *outputEmpty = nullptr;
  Operation *outputFill = nullptr;
  Operation *outputFillConstant = nullptr;
};

// Records the one constant producer found among a pair of layer inputs.
struct SingleConstantInput {
  Operation *constant = nullptr;
};

// Carries expanded input and weight operands after identifying the constant side.
struct ExpandedInputAndWeight {
  Operation *inputExpand = nullptr;
  Operation *weightExpand = nullptr;
  Operation *weightConstant = nullptr;
};

// Matches the broadcast-based output initializer used to apply a bias tensor.
inline std::optional<BiasOutputInit> matchBroadcastOutputInit(Value value) {
  auto outputBroadcast = extractor_utils::defOpAs<linalg::BroadcastOp>(value);
  if (!outputBroadcast)
    return std::nullopt;

  Operation *broadcastOp = outputBroadcast.getOperation();
  if (!extractor_utils::hasOperands(broadcastOp, 2))
    return std::nullopt;

  if (!extractor_utils::hasInputs(broadcastOp, 1))
    return std::nullopt;

  auto biasConstant =
      extractor_utils::defOpAs<arith::ConstantOp>(broadcastOp, 0);
  if (!biasConstant)
    return std::nullopt;

  auto outputEmpty = extractor_utils::defOpAs<tensor::EmptyOp>(broadcastOp, 1);
  if (!outputEmpty)
    return std::nullopt;

  return BiasOutputInit{biasConstant.getOperation(), outputEmpty.getOperation(),
                        broadcastOp};
}

// Matches a broadcast output initializer through a checked operand lookup.
inline std::optional<BiasOutputInit>
matchBroadcastOutputInit(Operation *op, unsigned operandIndex) {
  if (!op || operandIndex >= op->getNumOperands())
    return std::nullopt;

  return matchBroadcastOutputInit(op->getOperand(operandIndex));
}

// Matches the fill-based output initializer used when a layer has no bias.
inline std::optional<FillOutputInit> matchFillOutputInit(Value value) {
  auto outputFill = extractor_utils::defOpAs<linalg::FillOp>(value);
  if (!outputFill)
    return std::nullopt;

  Operation *fillOp = outputFill.getOperation();
  if (!extractor_utils::hasOperands(fillOp, 2))
    return std::nullopt;

  if (!extractor_utils::hasInputs(fillOp, 1))
    return std::nullopt;

  auto fillConstant = extractor_utils::defOpAs<arith::ConstantOp>(fillOp, 0);
  if (!fillConstant)
    return std::nullopt;

  auto outputEmpty = extractor_utils::defOpAs<tensor::EmptyOp>(fillOp, 1);
  if (!outputEmpty)
    return std::nullopt;

  return FillOutputInit{outputEmpty.getOperation(), fillOp,
                        fillConstant.getOperation()};
}

// Matches a fill output initializer through a checked operand lookup.
inline std::optional<FillOutputInit>
matchFillOutputInit(Operation *op, unsigned operandIndex) {
  if (!op || operandIndex >= op->getNumOperands())
    return std::nullopt;

  return matchFillOutputInit(op->getOperand(operandIndex));
}

// Returns the first constant producer found in two possible weight slots.
inline Operation *findConstantInput(Operation *op, unsigned firstIndex,
                                    unsigned secondIndex) {
  if (!op)
    return nullptr;

  if (auto firstConstant =
          extractor_utils::defOpAs<arith::ConstantOp>(op, firstIndex))
    return firstConstant.getOperation();

  if (auto secondConstant =
          extractor_utils::defOpAs<arith::ConstantOp>(op, secondIndex))
    return secondConstant.getOperation();

  return nullptr;
}

// Matches a two-input op only when exactly one candidate operand is constant.
inline std::optional<SingleConstantInput>
matchSingleConstantInputPair(Operation *op, unsigned firstIndex = 0,
                             unsigned secondIndex = 1) {
  if (!op || firstIndex >= op->getNumOperands() ||
      secondIndex >= op->getNumOperands())
    return std::nullopt;

  auto firstConstant =
      extractor_utils::defOpAs<arith::ConstantOp>(op, firstIndex);
  auto secondConstant =
      extractor_utils::defOpAs<arith::ConstantOp>(op, secondIndex);
  if (static_cast<bool>(firstConstant) == static_cast<bool>(secondConstant))
    return std::nullopt;

  Value input = firstConstant ? op->getOperand(secondIndex)
                              : op->getOperand(firstIndex);
  if (extractor_utils::defOpAs<arith::ConstantOp>(input))
    return std::nullopt;

  Operation *constant = firstConstant ? firstConstant.getOperation()
                                      : secondConstant.getOperation();
  return SingleConstantInput{constant};
}

// Matches expanded input/weight operands while accepting either operand order.
inline std::optional<ExpandedInputAndWeight>
matchExpandedInputAndWeight(Operation *op, unsigned firstIndex = 0,
                            unsigned secondIndex = 1) {
  auto firstExpand =
      extractor_utils::defOpAs<tensor::ExpandShapeOp>(op, firstIndex);
  if (!firstExpand)
    return std::nullopt;

  if (!extractor_utils::hasOperands(firstExpand.getOperation(), 1))
    return std::nullopt;

  auto secondExpand =
      extractor_utils::defOpAs<tensor::ExpandShapeOp>(op, secondIndex);
  if (!secondExpand)
    return std::nullopt;

  if (!extractor_utils::hasOperands(secondExpand.getOperation(), 1))
    return std::nullopt;

  bool firstIsConstant =
      static_cast<bool>(extractor_utils::defOpAs<arith::ConstantOp>(
          firstExpand.getSrc()));
  bool secondIsConstant =
      static_cast<bool>(extractor_utils::defOpAs<arith::ConstantOp>(
          secondExpand.getSrc()));
  if (firstIsConstant == secondIsConstant)
    return std::nullopt;

  auto inputExpand = firstIsConstant ? secondExpand : firstExpand;
  auto weightExpand = firstIsConstant ? firstExpand : secondExpand;
  auto weightConstant =
      extractor_utils::defOpAs<arith::ConstantOp>(weightExpand.getSrc());
  if (!weightConstant)
    return std::nullopt;

  return ExpandedInputAndWeight{inputExpand.getOperation(),
                                weightExpand.getOperation(),
                                weightConstant.getOperation()};
}

// Walks a function until a matcher finds the next extractable operation.
template <typename MatchFn>
auto findNextMatch(func::FuncOp func, MatchFn matchFn)
    -> decltype(matchFn(static_cast<Operation *>(nullptr))) {
  decltype(matchFn(static_cast<Operation *>(nullptr))) match;
  func.getBody().walk([&](Operation *op) -> WalkResult {
    match = matchFn(op);
    if (match)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return match;
}

// Repeatedly rewrites matches, rewalking after each mutation to avoid stale ops.
template <typename MatchFn, typename RewriteFn>
void extractAllMatches(func::FuncOp func, RewriterBase &rewriter,
                       MatchFn matchFn, RewriteFn rewriteFn) {
  auto match = findNextMatch(func, matchFn);
  while (match) {
    rewriteFn(*match, rewriter);
    // Re-scan because each extraction mutates the function body.
    match = findNextMatch(func, matchFn);
  }
}

} // namespace extractor_impl
} // namespace analog
} // namespace mlir

#endif // ANALOG_MLIR_DIALECT_ANALOG_TRANSFORMS_EXTRACTORS_EXTRACTORIMPLEMENTATIONUTILS_H
