#include "analog-mlir/Dialect/Analog/Transforms/assemblers/TaskGraphAssemblyStep.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include <memory>
#include <optional>
#include <string>

namespace {

using Segment = llvm::SmallVector<mlir::Operation *>;

bool containsOp(llvm::ArrayRef<mlir::Operation *> ops, mlir::Operation *op) {
  return llvm::is_contained(ops, op);
}

bool isOperationInsideSegment(mlir::Operation *op,
                              llvm::ArrayRef<mlir::Operation *> segment) {
  if (!op)
    return false;

  for (mlir::Operation *current = op; current; current = current->getParentOp()) {
    if (containsOp(segment, current))
      return true;
  }

  return false;
}

bool isValueInsideSegment(mlir::Value value,
                          llvm::ArrayRef<mlir::Operation *> segment) {
  if (mlir::Operation *definingOp = value.getDefiningOp())
    return isOperationInsideSegment(definingOp, segment);

  auto blockArgument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArgument)
    return false;

  return isOperationInsideSegment(blockArgument.getOwner()->getParentOp(),
                                  segment);
}

void appendUniqueValue(llvm::SmallVectorImpl<mlir::Value> &values,
                       mlir::Value value) {
  if (llvm::is_contained(values, value))
    return;
  values.push_back(value);
}

std::string makeUniqueBoundaryFunctionName(mlir::ModuleOp module) {
  unsigned functionIndex = 0;
  std::string functionName = "forward_task_" + std::to_string(functionIndex);
  while (module.lookupSymbol(functionName)) {
    ++functionIndex;
    functionName = "forward_task_" + std::to_string(functionIndex);
  }
  return functionName;
}

llvm::SmallVector<Segment> collectBoundarySegments(mlir::func::FuncOp func) {
  llvm::SmallVector<Segment> segments;
  mlir::Block &entryBlock = func.getBody().front();
  Segment currentSegment;

  for (mlir::Operation &op : entryBlock) {
    if (llvm::isa<mlir::func::CallOp>(op) ||
        llvm::isa<mlir::func::ReturnOp>(op)) {
      if (!currentSegment.empty()) {
        segments.push_back(std::move(currentSegment));
        currentSegment.clear();
      }
      continue;
    }

    currentSegment.push_back(&op);
  }

  if (!currentSegment.empty())
    segments.push_back(std::move(currentSegment));

  return segments;
}

bool isMovableBoundaryOp(mlir::Operation *op) {
  return llvm::isa<mlir::arith::ConstantOp, mlir::tensor::EmptyOp>(op);
}

bool isCloneableBoundaryValue(mlir::Value value) {
  mlir::Operation *definingOp = value.getDefiningOp();
  return definingOp && isMovableBoundaryOp(definingOp);
}

bool isMovableBoundarySegment(llvm::ArrayRef<mlir::Operation *> segment) {
  if (segment.empty())
    return false;

  return llvm::all_of(segment, [](mlir::Operation *op) {
    return isMovableBoundaryOp(op);
  });
}

std::optional<unsigned>
findContainingSegmentIndex(mlir::Operation *op,
                           llvm::ArrayRef<Segment> segments) {
  for (unsigned i = 0; i < segments.size(); ++i) {
    if (isOperationInsideSegment(op, segments[i]))
      return i;
  }

  return std::nullopt;
}

std::optional<unsigned>
findUniqueLaterConsumerSegmentIndex(llvm::ArrayRef<Segment> segments,
                                    unsigned producerIndex) {
  if (producerIndex >= segments.size())
    return std::nullopt;

  llvm::SetVector<unsigned> consumerIndices;
  llvm::ArrayRef<mlir::Operation *> producerSegment = segments[producerIndex];

  for (mlir::Operation *op : producerSegment) {
    for (mlir::Value result : op->getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        mlir::Operation *owner = use.getOwner();
        if (isOperationInsideSegment(owner, producerSegment))
          continue;

        if (llvm::isa<mlir::func::CallOp, mlir::func::ReturnOp>(owner))
          return std::nullopt;

        std::optional<unsigned> consumerIndex =
            findContainingSegmentIndex(owner, segments);
        if (!consumerIndex || *consumerIndex <= producerIndex)
          return std::nullopt;

        consumerIndices.insert(*consumerIndex);
        if (consumerIndices.size() > 1)
          return std::nullopt;
      }
    }
  }

  if (consumerIndices.empty())
    return std::nullopt;

  return *consumerIndices.begin();
}

bool sinkMovableBoundaryDependencies(mlir::func::FuncOp func) {
  bool changed = false;

  while (true) {
    llvm::SmallVector<Segment> segments = collectBoundarySegments(func);
    bool movedAnySegment = false;

    for (unsigned producerIndex = 0; producerIndex < segments.size();
         ++producerIndex) {
      llvm::ArrayRef<mlir::Operation *> producerSegment = segments[producerIndex];
      if (!isMovableBoundarySegment(producerSegment))
        continue;

      std::optional<unsigned> consumerIndex =
          findUniqueLaterConsumerSegmentIndex(segments, producerIndex);
      if (!consumerIndex)
        continue;

      mlir::Operation *anchorOp = segments[*consumerIndex].front();
      for (mlir::Operation *op : producerSegment)
        op->moveBefore(anchorOp);

      movedAnySegment = true;
      changed = true;
      break;
    }

    if (!movedAnySegment)
      return changed;
  }
}

bool shouldCloneExternalValue(mlir::Value value,
                              llvm::ArrayRef<mlir::Operation *> segment) {
  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp || isOperationInsideSegment(definingOp, segment))
    return false;

  return isCloneableBoundaryValue(value);
}

void collectExternalValues(llvm::ArrayRef<mlir::Operation *> segment,
                           llvm::SmallVectorImpl<mlir::Value> &externalValues) {
  externalValues.clear();

  for (mlir::Operation *op : segment) {
    op->walk([&](mlir::Operation *nestedOp) {
      for (mlir::Value operand : nestedOp->getOperands()) {
        if (isValueInsideSegment(operand, segment))
          continue;
        appendUniqueValue(externalValues, operand);
      }
    });
  }
}

void collectCloneOpsForValue(mlir::Value value,
                             llvm::ArrayRef<mlir::Operation *> segment,
                             llvm::SetVector<mlir::Operation *> &cloneOps,
                             llvm::SmallVectorImpl<mlir::Value> &inputs) {
  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp || isOperationInsideSegment(definingOp, segment)) {
    appendUniqueValue(inputs, value);
    return;
  }

  if (!shouldCloneExternalValue(value, segment)) {
    appendUniqueValue(inputs, value);
    return;
  }

  for (mlir::Value operand : definingOp->getOperands())
    collectCloneOpsForValue(operand, segment, cloneOps, inputs);

  cloneOps.insert(definingOp);
}

void collectSegmentInputsAndCloneOps(
    llvm::ArrayRef<mlir::Operation *> segment,
    llvm::SmallVectorImpl<mlir::Value> &inputs,
    llvm::SetVector<mlir::Operation *> &cloneOps) {
  inputs.clear();
  cloneOps.clear();

  llvm::SmallVector<mlir::Value> externalValues;
  collectExternalValues(segment, externalValues);
  for (mlir::Value value : externalValues)
    collectCloneOpsForValue(value, segment, cloneOps, inputs);
}

void appendUniqueOutput(llvm::SmallVectorImpl<mlir::Value> &outputs,
                        mlir::Value output) {
  if (llvm::is_contained(outputs, output))
    return;
  outputs.push_back(output);
}

bool escapesSegment(mlir::Value result,
                    llvm::ArrayRef<mlir::Operation *> segment) {
  return llvm::any_of(result.getUses(), [&](mlir::OpOperand &use) {
    return !isOperationInsideSegment(use.getOwner(), segment);
  });
}

mlir::LogicalResult
materializeCloneableEscapingOutputs(mlir::func::FuncOp parentFunc,
                                    llvm::ArrayRef<mlir::Operation *> segment,
                                    mlir::RewriterBase &rewriter) {
  llvm::SmallVector<Segment> segments = collectBoundarySegments(parentFunc);

  for (mlir::Operation *op : segment) {
    if (!isMovableBoundaryOp(op))
      continue;

    llvm::DenseMap<unsigned, mlir::Operation *> cloneByConsumerIndex;
    for (mlir::Value result : op->getResults()) {
      unsigned resultNumber = llvm::cast<mlir::OpResult>(result).getResultNumber();
      llvm::SmallVector<mlir::OpOperand *> externalUses;
      for (mlir::OpOperand &use : result.getUses()) {
        if (!isOperationInsideSegment(use.getOwner(), segment))
          externalUses.push_back(&use);
      }

      for (mlir::OpOperand *use : externalUses) {
        mlir::Operation *owner = use->getOwner();
        std::optional<unsigned> consumerIndex =
            findContainingSegmentIndex(owner, segments);
        if (!consumerIndex) {
          op->emitError("expected cloneable boundary value to be consumed by "
                        "a later outlined segment");
          return mlir::failure();
        }

        mlir::Operation *&consumerClone = cloneByConsumerIndex[*consumerIndex];
        if (!consumerClone) {
          rewriter.setInsertionPoint(segments[*consumerIndex].front());
          consumerClone = rewriter.clone(*op);
        }

        use->set(consumerClone->getResult(resultNumber));
      }
    }
  }

  return mlir::success();
}

void collectSegmentOutputs(llvm::ArrayRef<mlir::Operation *> segment,
                           llvm::SmallVectorImpl<mlir::Value> &outputs) {
  outputs.clear();

  for (mlir::Operation *op : segment) {
    for (mlir::Value result : op->getResults()) {
      if (escapesSegment(result, segment) && !isCloneableBoundaryValue(result))
        appendUniqueOutput(outputs, result);
    }
  }
}

mlir::LogicalResult extractSegmentToFunction(mlir::func::FuncOp parentFunc,
                                             llvm::ArrayRef<mlir::Operation *> segment,
                                             mlir::RewriterBase &rewriter) {
  if (segment.empty())
    return mlir::success();

  auto module = parentFunc->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return mlir::success();

  if (failed(materializeCloneableEscapingOutputs(parentFunc, segment, rewriter)))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> inputs;
  llvm::SetVector<mlir::Operation *> cloneOps;
  llvm::SmallVector<mlir::Value> outputs;
  collectSegmentInputsAndCloneOps(segment, inputs, cloneOps);
  collectSegmentOutputs(segment, outputs);

  llvm::SmallVector<mlir::Type> inputTypes;
  llvm::SmallVector<mlir::Type> outputTypes;
  for (mlir::Value input : inputs)
    inputTypes.push_back(input.getType());
  for (mlir::Value output : outputs)
    outputTypes.push_back(output.getType());

  std::string functionName = makeUniqueBoundaryFunctionName(module);
  auto functionType = rewriter.getFunctionType(inputTypes, outputTypes);

  rewriter.setInsertionPointToEnd(module.getBody());
  auto extractedFunc = rewriter.create<mlir::func::FuncOp>(
      segment.front()->getLoc(), functionName, functionType);
  extractedFunc.setPrivate();

  mlir::Block *entryBlock = extractedFunc.addEntryBlock();
  mlir::IRMapping mapping;
  for (unsigned i = 0; i < inputs.size(); ++i)
    mapping.map(inputs[i], entryBlock->getArgument(i));

  rewriter.setInsertionPointToStart(entryBlock);
  for (mlir::Operation *cloneOp : cloneOps)
    rewriter.clone(*cloneOp, mapping);
  for (mlir::Operation *op : segment)
    rewriter.clone(*op, mapping);

  llvm::SmallVector<mlir::Value> mappedOutputs;
  for (mlir::Value output : outputs)
    mappedOutputs.push_back(mapping.lookupOrDefault(output));
  rewriter.create<mlir::func::ReturnOp>(segment.back()->getLoc(), mappedOutputs);

  rewriter.setInsertionPoint(segment.front());
  auto call = rewriter.create<mlir::func::CallOp>(
      segment.front()->getLoc(), extractedFunc.getSymName(), outputTypes, inputs);

  for (unsigned i = 0; i < outputs.size(); ++i) {
    mlir::Value output = outputs[i];
    output.replaceUsesWithIf(call.getResult(i), [&](mlir::OpOperand &use) {
      return !isOperationInsideSegment(use.getOwner(), segment);
    });
  }

  for (auto it = segment.rbegin(); it != segment.rend(); ++it)
    rewriter.eraseOp(*it);

  return mlir::success();
}

class ForwardTaskOutliner final : public mlir::analog::TaskGraphAssemblyStep {
public:
  mlir::StringRef getName() const final { return "ForwardTaskOutliner"; }

  mlir::LogicalResult assemble(mlir::ModuleOp,
                               mlir::func::FuncOp forward) const final {
    if (!forward.getBody().hasOneBlock()) {
      forward.emitError("expected forward to have a single block");
      return mlir::failure();
    }

    mlir::IRRewriter rewriter(forward.getContext());
    while (true) {
      sinkMovableBoundaryDependencies(forward);
      llvm::SmallVector<Segment> segments = collectBoundarySegments(forward);
      if (segments.empty())
        return mlir::success();

      if (failed(extractSegmentToFunction(forward, segments.front(), rewriter)))
        return mlir::failure();
    }
  }
};

} // namespace

namespace mlir {
namespace analog {

void registerForwardTaskOutliner(TaskGraphAssemblySteps &steps) {
  steps.push_back(std::make_unique<ForwardTaskOutliner>());
}

} // namespace analog
} // namespace mlir
