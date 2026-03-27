#include "analog-mlir/Dialect/Analog/Transforms/IdentifyRecurrentPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

constexpr StringLiteral kRecurrentPatternAttr = "analog.recurrent_pattern";
constexpr StringLiteral kRecurrentActivationAttr =
    "analog.recurrent_activation";
constexpr StringLiteral kRecurrentInputSizeAttr =
    "analog.recurrent_input_size";
constexpr StringLiteral kRecurrentHiddenSizeAttr =
    "analog.recurrent_hidden_size";
constexpr StringLiteral kRecurrentStepsAttr = "analog.recurrent_steps";

struct RecurrentPatternMatch {
  StringRef kind;
  Operation *anchor = nullptr;
  SmallVector<Operation *> matchedOps;
  Value currentInput;
  Value hiddenInput;
  Value result;
  int64_t inputSize = -1;
  int64_t hiddenSize = -1;
  int64_t steps = -1;
  StringRef activation;
};

class RecurrentPatternMatcher {
public:
  virtual ~RecurrentPatternMatcher() = default;
  virtual void match(func::FuncOp func, DenseSet<Operation *> &claimedOps,
                     SmallVectorImpl<RecurrentPatternMatch> &matches) const = 0;
};

//===----------------------------------------------------------------------===//
// Generic Match Helpers
//===----------------------------------------------------------------------===//

// Returns whether the generic is a simple elementwise floating-point add.
static bool isPointwiseAddGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1)
    return false;

  Block &body = generic.getRegion().front();
  auto it = body.begin();
  auto addOp = dyn_cast<arith::AddFOp>(*it++);
  if (!addOp)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(*it++);
  if (!yieldOp || it != body.end())
    return false;
  return yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == addOp.getResult();
}

// Returns whether the generic is a simple elementwise tanh.
static bool isTanhGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;

  Block &body = generic.getRegion().front();
  auto it = body.begin();
  auto tanhOp = dyn_cast<math::TanhOp>(*it++);
  if (!tanhOp)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(*it++);
  if (!yieldOp || it != body.end())
    return false;
  return yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == tanhOp.getResult();
}

// Returns the constant feeding a canonical weight transpose.
static arith::ConstantOp getRank2ConstantThroughTranspose(Value value) {
  auto transpose = value.getDefiningOp<linalg::TransposeOp>();
  if (!transpose)
    return {};
  auto permutation = transpose.getPermutation();
  if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
    return {};
  return transpose.getInput().getDefiningOp<arith::ConstantOp>();
}

// Returns whether any operation in `ops` was already claimed by a previous
// recurrent-pattern match.
static bool isClaimed(ArrayRef<Operation *> ops, const DenseSet<Operation *> &claimed) {
  return llvm::any_of(ops, [&](Operation *op) { return claimed.contains(op); });
}

// Marks each operation as claimed so later matchers skip overlapping matches.
static void claimMatchedOps(ArrayRef<Operation *> ops, DenseSet<Operation *> &claimed) {
  for (Operation *op : ops)
    claimed.insert(op);
}

// Walks up the parent chain to the operation directly owned by the function.
static Operation *findTopLevelOwner(Operation *op) {
  Operation *top = op;
  while (top && !isa<func::FuncOp>(top->getParentOp()))
    top = top->getParentOp();
  return top;
}

// Collects the top-level dependency segment needed to compute `rootValue`.
static SmallVector<Operation *> collectTopLevelDependencySegment(Value rootValue) {
  DenseSet<Operation *> owners;
  SmallVector<Value> worklist = {rootValue};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *producer = value.getDefiningOp();
    if (!producer || isa<arith::ConstantOp>(producer))
      continue;
    Operation *top = findTopLevelOwner(producer);
    if (!top || owners.contains(top))
      continue;
    owners.insert(top);
    top->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        worklist.push_back(operand);
    });
  }

  SmallVector<Operation *> segment;
  Block *block = nullptr;
  for (Operation *op : owners) {
    if (!block)
      block = op->getBlock();
  }
  if (!block)
    return segment;
  for (Operation &op : *block) {
    if (owners.contains(&op))
      segment.push_back(&op);
  }
  return segment;
}

// Moves a contiguous top-level segment into the target block while preserving
// the original operation order.
static void moveSegmentIntoBlock(ArrayRef<Operation *> segment, Block *targetBlock) {
  for (Operation *op : segment)
    op->moveBefore(targetBlock, targetBlock->end());
}

//===----------------------------------------------------------------------===//
// RNN Cell Matcher
//===----------------------------------------------------------------------===//

class RNNCellMatcher final : public RecurrentPatternMatcher {
public:
  void match(func::FuncOp func, DenseSet<Operation *> &claimedOps,
             SmallVectorImpl<RecurrentPatternMatch> &matches) const override {
    func.walk([&](linalg::GenericOp activationOp) {
      if (!isTanhGeneric(activationOp))
        return;
      if (claimedOps.contains(activationOp.getOperation()))
        return;

      Value activationInput = activationOp.getDpsInputOperand(0)->get();
      auto mergeAdd = activationInput.getDefiningOp<linalg::GenericOp>();
      if (!isPointwiseAddGeneric(mergeAdd))
        return;

      Value lhs = mergeAdd.getDpsInputOperand(0)->get();
      Value rhs = mergeAdd.getDpsInputOperand(1)->get();
      auto lhsBiasAdd = lhs.getDefiningOp<linalg::GenericOp>();
      auto rhsBiasAdd = rhs.getDefiningOp<linalg::GenericOp>();
      if (!isPointwiseAddGeneric(lhsBiasAdd) || !isPointwiseAddGeneric(rhsBiasAdd))
        return;

      auto lhsMatmul = lhsBiasAdd.getDpsInputOperand(0)->get().getDefiningOp<linalg::MatmulOp>();
      auto rhsMatmul = rhsBiasAdd.getDpsInputOperand(0)->get().getDefiningOp<linalg::MatmulOp>();
      if (!lhsMatmul || !rhsMatmul)
        return;

      Value lhsBias = lhsBiasAdd.getDpsInputOperand(1)->get();
      Value rhsBias = rhsBiasAdd.getDpsInputOperand(1)->get();
      auto lhsBiasConst = lhsBias.getDefiningOp<arith::ConstantOp>();
      auto rhsBiasConst = rhsBias.getDefiningOp<arith::ConstantOp>();
      if (!lhsBiasConst || !rhsBiasConst)
        return;

      auto lhsWeightConst = getRank2ConstantThroughTranspose(lhsMatmul.getInputs()[1]);
      auto rhsWeightConst = getRank2ConstantThroughTranspose(rhsMatmul.getInputs()[1]);
      if (!lhsWeightConst || !rhsWeightConst)
        return;

      Value lhsInput = lhsMatmul.getInputs()[0];
      Value rhsInput = rhsMatmul.getInputs()[0];
      auto lhsInputTy = dyn_cast<RankedTensorType>(lhsInput.getType());
      auto rhsInputTy = dyn_cast<RankedTensorType>(rhsInput.getType());
      auto outputTy = dyn_cast<RankedTensorType>(activationOp.getResult(0).getType());
      auto lhsWeightTy = dyn_cast<RankedTensorType>(lhsWeightConst.getType());
      auto rhsWeightTy = dyn_cast<RankedTensorType>(rhsWeightConst.getType());
      auto lhsBiasTy = dyn_cast<RankedTensorType>(lhsBias.getType());
      auto rhsBiasTy = dyn_cast<RankedTensorType>(rhsBias.getType());
      if (!lhsInputTy || !rhsInputTy || !outputTy || !lhsWeightTy || !rhsWeightTy ||
          !lhsBiasTy || !rhsBiasTy)
        return;
      if (!lhsInputTy.hasStaticShape() || !rhsInputTy.hasStaticShape() ||
          !outputTy.hasStaticShape())
        return;
      if (lhsInputTy.getRank() != 2 || rhsInputTy.getRank() != 2 ||
          outputTy.getRank() != 2 || outputTy.getShape()[0] != 1)
        return;
      if (lhsBiasTy.getRank() != 1 || rhsBiasTy.getRank() != 1)
        return;

      int64_t hiddenSize = outputTy.getShape()[1];
      Value hiddenInput = lhsInput;
      Value currentInput = rhsInput;
      auto hiddenInputTy = lhsInputTy;
      auto currentInputTy = rhsInputTy;
      auto hiddenWeightTy = lhsWeightTy;
      auto inputWeightTy = rhsWeightTy;
      if (lhsInputTy.getShape()[1] != hiddenSize && rhsInputTy.getShape()[1] == hiddenSize) {
        hiddenInput = rhsInput;
        currentInput = lhsInput;
        hiddenInputTy = rhsInputTy;
        currentInputTy = lhsInputTy;
        hiddenWeightTy = rhsWeightTy;
        inputWeightTy = lhsWeightTy;
      }

      if (hiddenInputTy.getShape()[0] != 1 || currentInputTy.getShape()[0] != 1)
        return;
      if (hiddenInputTy.getShape()[1] != hiddenSize)
        return;
      if (hiddenWeightTy.getShape()[0] != hiddenSize || inputWeightTy.getShape()[0] != hiddenSize)
        return;
      if (hiddenWeightTy.getShape()[1] != hiddenSize)
        return;
      if (lhsBiasTy.getShape()[0] != hiddenSize || rhsBiasTy.getShape()[0] != hiddenSize)
        return;

      SmallVector<Operation *> ops =
          collectTopLevelDependencySegment(activationOp.getResult(0));
      if (isClaimed(ops, claimedOps))
        return;

      RecurrentPatternMatch match;
      match.kind = "rnn_cell";
      match.anchor = activationOp.getOperation();
      match.matchedOps = ops;
      match.currentInput = currentInput;
      match.hiddenInput = hiddenInput;
      match.result = activationOp.getResult(0);
      match.inputSize = currentInputTy.getShape()[1];
      match.hiddenSize = hiddenSize;
      match.activation = "tanh";
      claimMatchedOps(ops, claimedOps);
      matches.push_back(std::move(match));
    });
  }
};

// Strips simple view-like reshapes and slices so recurrent edges can be matched
// through canonicalized sequence plumbing.
static Value stripSimpleViewLike(Value value) {
  while (true) {
    if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
      value = expand.getSrc();
      continue;
    }
    if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
      value = slice.getSource();
      continue;
    }
    break;
  }
  return value;
}

//===----------------------------------------------------------------------===//
// RNN Sequence Matcher
//===----------------------------------------------------------------------===//

class RNNSequenceMatcher final : public RecurrentPatternMatcher {
public:
  void match(func::FuncOp func, DenseSet<Operation *> &claimedOps,
             SmallVectorImpl<RecurrentPatternMatch> &matches) const override {
    if (func.getNumArguments() != 2)
      return;
    auto sequenceInputTy =
        dyn_cast<RankedTensorType>(func.getArgument(0).getType());
    auto initialHiddenTy =
        dyn_cast<RankedTensorType>(func.getArgument(1).getType());
    if (!sequenceInputTy || !initialHiddenTy ||
        !sequenceInputTy.hasStaticShape() || !initialHiddenTy.hasStaticShape() ||
        sequenceInputTy.getRank() != 3 || initialHiddenTy.getRank() != 3 ||
        sequenceInputTy.getElementType() !=
            initialHiddenTy.getElementType() ||
        !sequenceInputTy.getElementType().isF32())
      return;

    SmallVector<linalg::GenericOp> tanhOps;
    func.walk([&](linalg::GenericOp generic) {
      if (isTanhGeneric(generic)) {
        auto resultTy =
            dyn_cast<RankedTensorType>(generic.getResult(0).getType());
        if (resultTy && resultTy.hasStaticShape() && resultTy.getRank() == 3)
          tanhOps.push_back(generic);
      }
    });
    if (tanhOps.size() < 2)
      return;

    llvm::sort(tanhOps, [](linalg::GenericOp lhs, linalg::GenericOp rhs) {
      return lhs->isBeforeInBlock(rhs.getOperation());
    });

    for (size_t i = 1; i < tanhOps.size(); ++i) {
      linalg::GenericOp prev = tanhOps[i - 1];
      linalg::GenericOp cur = tanhOps[i];
      Value curInput = cur.getDpsInputOperand(0)->get();
      auto mergeAdd = curInput.getDefiningOp<linalg::GenericOp>();
      if (!isPointwiseAddGeneric(mergeAdd))
        continue;

      SmallVector<Value> branchInputs = {mergeAdd.getDpsInputOperand(0)->get(),
                                         mergeAdd.getDpsInputOperand(1)->get()};
      bool foundRecurrentEdge = false;
      for (Value branchInput : branchInputs) {
        Value normalized = stripSimpleViewLike(branchInput);
        auto branchBiasAdd = normalized.getDefiningOp<linalg::GenericOp>();
        if (isPointwiseAddGeneric(branchBiasAdd))
          normalized = branchBiasAdd.getDpsInputOperand(0)->get();
        auto branchMatmul = normalized.getDefiningOp<linalg::MatmulOp>();
        if (!branchMatmul)
          continue;
        if (stripSimpleViewLike(branchMatmul.getInputs()[0]) == prev.getResult(0)) {
          foundRecurrentEdge = true;
          break;
        }
      }
      if (!foundRecurrentEdge)
        continue;

      RecurrentPatternMatch seq;
      seq.kind = "rnn";
      seq.anchor = func.getOperation();
      if (auto returnOp =
              dyn_cast<func::ReturnOp>(&func.getBody().front().back())) {
        if (returnOp.getNumOperands() == 1) {
          seq.result = returnOp.getOperand(0);
          seq.matchedOps = collectTopLevelDependencySegment(returnOp.getOperand(0));
        }
      }
      if (!seq.result || seq.matchedOps.empty() ||
          isClaimed(seq.matchedOps, claimedOps))
        continue;

      seq.steps = static_cast<int64_t>(tanhOps.size());
      seq.hiddenSize = initialHiddenTy.getShape()[2];
      seq.inputSize = sequenceInputTy.getShape()[2];
      seq.activation = "tanh";
      claimMatchedOps(seq.matchedOps, claimedOps);
      matches.push_back(std::move(seq));
      break;
    }
  }
};

// Builds the recurrent-pattern registry in descending match specificity.
static SmallVector<std::unique_ptr<RecurrentPatternMatcher>>
buildRecurrentPatternRegistry() {
  SmallVector<std::unique_ptr<RecurrentPatternMatcher>> matchers;
  matchers.push_back(std::make_unique<RNNSequenceMatcher>());
  matchers.push_back(std::make_unique<RNNCellMatcher>());
  return matchers;
}

// Annotates the newly created isolation block with the recurrent metadata
// produced by the matcher.
static void annotateOperation(Operation *op, RecurrentPatternMatch &match) {
  if (!op)
    return;
  auto *ctx = op->getContext();
  op->setAttr(kRecurrentPatternAttr, StringAttr::get(ctx, match.kind));
  if (!match.activation.empty())
    op->setAttr(kRecurrentActivationAttr, StringAttr::get(ctx, match.activation));
  if (match.inputSize >= 0)
    op->setAttr(kRecurrentInputSizeAttr,
                IntegerAttr::get(IntegerType::get(ctx, 64), match.inputSize));
  if (match.hiddenSize >= 0)
    op->setAttr(kRecurrentHiddenSizeAttr,
                IntegerAttr::get(IntegerType::get(ctx, 64), match.hiddenSize));
  if (match.steps >= 0)
    op->setAttr(kRecurrentStepsAttr,
                IntegerAttr::get(IntegerType::get(ctx, 64), match.steps));
}

// Wraps the matched top-level segment in an execute_region so recurrent
// boundaries remain visible in the IR for later passes and debugging.
static void wrapMatchInExecuteRegion(RecurrentPatternMatch &match) {
  if (match.matchedOps.empty() || !match.result)
    return;

  Operation *first = match.matchedOps.front();
  if (!first || !first->getBlock())
    return;

  OpBuilder builder(first);
  auto exec = builder.create<scf::ExecuteRegionOp>(first->getLoc(),
                                                   TypeRange{match.result.getType()});
  annotateOperation(exec.getOperation(), match);

  Block *body = builder.createBlock(&exec.getRegion());
  moveSegmentIntoBlock(match.matchedOps, body);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  bodyBuilder.create<scf::YieldOp>(first->getLoc(), ValueRange{match.result});

  Value oldResult = match.result;
  Value newResult = exec.getResult(0);
  oldResult.replaceUsesWithIf(newResult, [&](OpOperand &use) {
    return use.getOwner()->getParentRegion() != &exec.getRegion();
  });
}

} // namespace

llvm::StringRef IdentifyRecurrentPatternsPass::getArgument() const {
  return "analog-identify-recurrent-patterns";
}

llvm::StringRef IdentifyRecurrentPatternsPass::getDescription() const {
  return "Identify and isolate recurrent cell and sequence patterns";
}

void IdentifyRecurrentPatternsPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
}

void IdentifyRecurrentPatternsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  DenseSet<Operation *> claimedOps;
  SmallVector<RecurrentPatternMatch> matches;
  auto matchers = buildRecurrentPatternRegistry();
  for (const auto &matcher : matchers)
    matcher->match(func, claimedOps, matches);
  for (RecurrentPatternMatch &match : matches)
    wrapMatchInExecuteRegion(match);
}

std::unique_ptr<mlir::Pass> createIdentifyRecurrentPatternsPass() {
  return std::make_unique<IdentifyRecurrentPatternsPass>();
}

} // namespace analog
} // namespace mlir
