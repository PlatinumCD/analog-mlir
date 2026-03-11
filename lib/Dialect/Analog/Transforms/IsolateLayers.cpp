#include "analog-mlir/Dialect/Analog/Transforms/IsolateLayers.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

using OpChain = SmallVector<Operation *>;
using OpChainList = SmallVector<OpChain>;
using TaggedOpChain = std::pair<OpChain, StringRef>;
using TaggedOpChainList = SmallVector<TaggedOpChain>;

constexpr StringLiteral kMatrixInitializationAttr = "analog-matrix-initialization";
constexpr StringLiteral kLinearRoutineAttr = "analog-linear-routine";
constexpr StringLiteral kConv2DRoutineAttr = "analog-conv2d-routine";
constexpr StringLiteral kWeightIdAttr = "weight-id";
constexpr StringLiteral kLayerIdAttr = "layer-id";
constexpr StringLiteral kOutputChannelAssemblyAttr = "analog.output_channel_assembly";
constexpr StringLiteral kSlidingWindowPatchAttr = "analog.sliding_window_patch";
constexpr StringLiteral kSlidingWindowBiasAddAttr = "analog.sliding_window_bias_add";
constexpr StringLiteral kMatrixInitializationPrefix = "analog_matrix_initialization_";
constexpr StringLiteral kLinearRoutinePrefix = "analog_linear_routine_";
constexpr StringLiteral kConv2DRoutinePrefix = "analog_conv2d_routine_";
constexpr StringLiteral kRewrittenConv2DOutputAttr = "analog.rewritten_conv2d_output";


// Walks up the parent chain until it finds the top-level operation
// directly owned by the surrounding function body.

static Operation *findTopLevelOwner(Operation *op) {
  Operation *top = op;
  while (top && !isa<func::FuncOp>(top->getParentOp()))
    top = top->getParentOp();
  return top;
}


// Returns whether every use of the producer stays within top-level owners
// that are already part of the candidate segment.
static bool allUsesStayWithinTopLevelOwners(
    Operation *producer, const DenseSet<Operation *> &segmentOwners) {
  for (Value result : producer->getResults()) {
    for (Operation *user : result.getUsers()) {
      Operation *userTop = findTopLevelOwner(user);
      if (!userTop || !segmentOwners.contains(userTop))
        return false;
    }
  }
  return true;
}


// Returns whether the top-level op is local setup that should move with a
// rewritten conv rather than become a separate layer dependency.
static bool isAbsorbableTopLevelSetupOp(Operation *op) {
  return isa<tensor::EmptyOp, tensor::ExpandShapeOp, linalg::FillOp,
             linalg::BroadcastOp>(op);
}


// Returns whether the op is a top-level conv assembly loop that builds a
// rank-4 tensor result, including the degenerate 1x1 spatial case.
static bool isTopLevelConvAssembly(Operation *op) {
  if (!op || !op->hasAttr(kOutputChannelAssemblyAttr))
    return false;

  auto forOp = dyn_cast<scf::ForOp>(op);
  if (!forOp || forOp.getNumResults() != 1)
    return false;

  auto resultTy = dyn_cast<RankedTensorType>(forOp.getResult(0).getType());
  return resultTy && resultTy.getRank() == 4;
}


// Returns whether the top-level op is part of a rewritten conv boundary,
// including degenerate 1x1 convs that are split across several top-level ops.
static bool isConvBoundaryTopLevelOp(Operation *op) {
  return isAbsorbableTopLevelSetupOp(op) || isTopLevelConvAssembly(op) ||
         op->hasAttr(kSlidingWindowPatchAttr) ||
         op->hasAttr(kSlidingWindowBiasAddAttr);
}


// Grows a top-level segment backward to absorb local setup ops that feed the
// anchor, including values used by nested ops inside the segment.
static OpChain buildClosedTopLevelSegment(Operation *anchor) {
  if (!anchor)
    return {};

  DenseSet<Operation *> segmentOwners;
  SmallVector<Operation *> worklist;
  segmentOwners.insert(anchor);
  worklist.push_back(anchor);

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    current->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer || isa<arith::ConstantOp>(producer))
          continue;

        Operation *producerTop = findTopLevelOwner(producer);
        if (!producerTop || producerTop == current ||
            producerTop->getBlock() != anchor->getBlock() ||
            segmentOwners.contains(producerTop)) {
          continue;
        }
        if (!isConvBoundaryTopLevelOp(producerTop))
          continue;

        if (!allUsesStayWithinTopLevelOwners(producerTop, segmentOwners))
          continue;

        segmentOwners.insert(producerTop);
        worklist.push_back(producerTop);
      }
    });
  }

  OpChain segment;
  for (Operation *op = anchor; op; op = op->getPrevNode()) {
    if (segmentOwners.contains(op))
      segment.push_back(op);
  }
  std::reverse(segment.begin(), segment.end());
  return segment;
}


// Collects every operation in the segment, including nested ops, so
// later analyses can reason about closure and escaping uses.

static DenseSet<Operation *> collectSegmentClosure(ArrayRef<Operation *> segment) {
  DenseSet<Operation *> inChain;
  for (Operation *cur : segment) {
    if (!cur)
      continue;
    inChain.insert(cur);
    cur->walk([&](Operation *nested) { inChain.insert(nested); });
  }
  return inChain;
}


// Finds the top-level op segments that initialize analog matrices from
// dense resource-backed constants.

static void collectMatrixInitializationChains(func::FuncOp func,
                                              OpChainList &matrixChains) {
  func.walk([&](analog::MatrixFromTensorOp op) {
    Value src = op.getOperand();
    auto cst = src.getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return;
    if (!isa<DenseResourceElementsAttr>(cst.getValue()))
      return;

    analog::MatrixPartitionOp partition;
    for (Operation *user : op->getUsers()) {
      auto candidate = dyn_cast<analog::MatrixPartitionOp>(user);
      if (!candidate)
        continue;
      if (candidate.getOperand() != op.getResult())
        continue;
      partition = candidate;
      break;
    }
    if (!partition)
      return;

    analog::ArrayMatrixPlaceOp place;
    for (Operation *user : partition->getUsers()) {
      auto candidate = dyn_cast<analog::ArrayMatrixPlaceOp>(user);
      if (!candidate)
        continue;
      if (candidate.getOperand(0) != partition.getResult())
        continue;
      place = candidate;
      break;
    }
    if (!place)
      return;

    Operation *endTop = findTopLevelOwner(place);
    if (!endTop)
      return;
    if (cst->getBlock() != endTop->getBlock())
      return;

    // Capture the smallest top-level segment that still represents "program
    // this matrix into the device" without force-moving the source constant.
    OpChain segment;
    // Only move the matrix-init routine ops/containers. Do not force-move the
    // dense constant because it may also feed non-analog ops (e.g. transpose).
    segment.push_back(op);
    if (partition.getOperation() != op.getOperation())
      segment.push_back(partition.getOperation());
    if (endTop != op.getOperation() && endTop != partition.getOperation())
      segment.push_back(endTop);
    matrixChains.push_back(std::move(segment));
  });
}


// Finds the op segments that implement one analog layer routine from
// vector materialization through tensor re-materialization.

static void collectLayerRoutineChains(func::FuncOp func,
                                      TaggedOpChainList &layerRoutineChains) {
  DenseSet<Operation *> seenTopLevelOwners;

  func.walk([&](Operation *op) {
    if (!op->hasAttr(kRewrittenConv2DOutputAttr))
      return;

    Operation *topLevelOwner = findTopLevelOwner(op);
    if (!topLevelOwner || !seenTopLevelOwners.insert(topLevelOwner).second)
      return;

    OpChain segment = buildClosedTopLevelSegment(topLevelOwner);
    if (!segment.empty()) {
      for (Operation *segmentOp : segment)
        seenTopLevelOwners.insert(segmentOp);
      layerRoutineChains.push_back(
          {std::move(segment), StringRef(kConv2DRoutineAttr)});
    }
  });

  func.walk([&](Operation *op) {
    if (!isTopLevelConvAssembly(op))
      return;

    Operation *topLevelOwner = findTopLevelOwner(op);
    if (!topLevelOwner || !seenTopLevelOwners.insert(topLevelOwner).second)
      return;

    OpChain segment = buildClosedTopLevelSegment(topLevelOwner);
    if (!segment.empty()) {
      for (Operation *segmentOp : segment)
        seenTopLevelOwners.insert(segmentOp);
      layerRoutineChains.push_back(
          {std::move(segment), StringRef(kConv2DRoutineAttr)});
    }
  });

  func.walk([&](analog::VectorFromTensorOp op) {
    Operation *startTop = findTopLevelOwner(op.getOperation());
    if (!startTop || !startTop->getBlock())
      return;
    if (startTop->hasAttr(kRewrittenConv2DOutputAttr))
      return;
    if (!seenTopLevelOwners.insert(startTop).second)
      return;

    // A layer routine starts at the vector materialization boundary and runs
    // until the first tensor result is materialized back into SSA form.
    Operation *endTop = nullptr;
    for (Operation *cur = startTop; cur; cur = cur->getNextNode()) {
      bool foundToTensor = false;
      cur->walk([&](bufferization::ToTensorOp) { foundToTensor = true; });
      if (foundToTensor) {
        endTop = cur;
        break;
      }
    }
    if (!endTop)
      return;

    OpChain chain;
    for (Operation *cur = startTop; cur; cur = cur->getNextNode()) {
      chain.push_back(cur);
      if (cur == endTop)
        break;
    }
    if (!chain.empty() && chain.back() == endTop)
      layerRoutineChains.push_back(
          {std::move(chain), StringRef(kLinearRoutineAttr)});
  });
}


// Returns whether the candidate region is the target region or is nested
// beneath it through parent operations.
static bool isRegionInsideRegion(Region *candidate, Region &target) {
  for (Region *region = candidate; region; ) {
    if (region == &target)
      return true;
    Operation *parent = region->getParentOp();
    region = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}


// Checks whether any value produced inside the segment is consumed by
// operations outside that segment.

static bool hasEscapingUses(ArrayRef<Operation *> segment) {
  DenseSet<Operation *> inChain = collectSegmentClosure(segment);
  for (Operation *cur : segment) {
    if (!cur)
      continue;
    bool escaping = false;
    cur->walk([&](Operation *nested) {
      for (Value res : nested->getResults()) {
        for (Operation *user : res.getUsers()) {
          if (!inChain.contains(user)) {
            escaping = true;
            return;
          }
        }
        if (escaping)
          return;
      }
    });
    if (escaping)
      return true;
  }
  return false;
}


// Collects the produced values whose uses extend beyond the outlined
// segment so wrapper regions can return them.

static void computeEscapingResults(ArrayRef<Operation *> segment,
                                   SmallVectorImpl<Value> &escapingResults) {
  DenseSet<Operation *> inChain = collectSegmentClosure(segment);

  for (Operation *cur : segment) {
    if (!cur)
      continue;
    cur->walk([&](Operation *nested) {
      for (Value res : nested->getResults()) {
        bool escapes = false;
        for (Operation *user : res.getUsers()) {
          if (!inChain.contains(user)) {
            escapes = true;
            break;
          }
        }
        if (escapes)
          escapingResults.push_back(res);
      }
    });
  }
}


// Finds arith constants used inside the execute region but defined
// outside of it.

static SmallVector<arith::ConstantOp>
collectExternalConstantOpsForRegion(scf::ExecuteRegionOp exec) {
  SmallVector<arith::ConstantOp> constants;
  DenseSet<Operation *> seen;
  Region &region = exec.getRegion();

  region.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto cst = operand.getDefiningOp<arith::ConstantOp>();
      if (!cst)
        continue;
      if (cst->getParentRegion() == &region)
        continue;
      if (!seen.insert(cst.getOperation()).second)
        continue;
      constants.push_back(cst);
    }
  });

  return constants;
}


// Returns whether the given operation is nested anywhere inside the
// target region.

static bool isOpInsideRegion(Operation *op, Region &region) {
  for (Region *r = op ? op->getParentRegion() : nullptr; r; ) {
    if (r == &region)
      return true;
    Operation *parent = r->getParentOp();
    r = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}


// Returns whether the value is defined by an operation or block argument
// whose owner already lives inside the target region.
static bool isValueDefinedInsideRegion(Value value, Region &region) {
  if (!value)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    return owner && isRegionInsideRegion(owner->getParent(), region);
  }

  return isOpInsideRegion(value.getDefiningOp(), region);
}


// Collects region operands that are defined outside the execute region so
// outlined layer helpers can take them as explicit function arguments.
static void collectExternalValuesForRegion(
    scf::ExecuteRegionOp exec, SmallVectorImpl<Value> &externalValues) {
  DenseSet<Value> seen;
  Region &region = exec.getRegion();

  region.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (isValueDefinedInsideRegion(operand, region))
        continue;
      if (!seen.insert(operand).second)
        continue;
      externalValues.push_back(operand);
    }
  });
}


// Checks whether all uses of the constant stay within the target
// region boundary.

static bool allUsesInsideRegion(arith::ConstantOp cst, Region &region) {
  for (Operation *user : cst->getUsers()) {
    if (!isOpInsideRegion(user, region))
      return false;
  }
  return true;
}


// Checks whether all uses of the operation stay within the target
// region boundary.
static bool allUsesInsideRegion(Operation *op, Region &region) {
  if (!op)
    return false;

  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!isOpInsideRegion(user, region))
        return false;
    }
  }
  return true;
}


// Rewrites only the uses of a value that occur within the target
// region.

static void replaceUsesInsideRegion(Value oldValue, Value newValue, Region &region) {
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return isOpInsideRegion(use.getOwner(), region);
  });
}


// Moves each top-level operation from the segment into the destination
// block while skipping null or detached entries.

static void moveSegmentIntoBlock(ArrayRef<Operation *> segment, Block *body) {
  for (Operation *cur : segment) {
    if (!cur || !cur->getBlock())
      continue;
    cur->moveBefore(body, body->end());
  }
}


// Creates the canonical body and exit blocks for a new execute region
// and adds them to the region.

static std::pair<Block *, Block *> createExecuteRegionBlocks(
    scf::ExecuteRegionOp exec) {
  Block *body = new Block();
  Block *exit = new Block();
  exec.getRegion().push_back(body);
  exec.getRegion().push_back(exit);
  return {body, exit};
}


// Collects execute regions carrying a specific tag so they can be
// rewritten after the walk is complete.

static SmallVector<scf::ExecuteRegionOp> collectTaggedExecuteRegions(
    func::FuncOp func, StringRef tag) {
  SmallVector<scf::ExecuteRegionOp> execs;
  func.walk([&](scf::ExecuteRegionOp exec) {
    if (exec->hasAttr(tag))
      execs.push_back(exec);
  });
  return execs;
}


// Extracts the numeric id stored under the execute-region tag
// attribute.

static std::optional<int64_t> getTaggedRegionId(scf::ExecuteRegionOp exec,
                                                StringRef tag) {
  auto attr = exec->getAttrOfType<IntegerAttr>(tag);
  if (!attr)
    return std::nullopt;
  return attr.getValue().getSExtValue();
}


// Returns the parent module that owns any helper functions outlined
// from the current function.

static ModuleOp getOutliningModule(func::FuncOp forward) {
  return forward ? forward->getParentOfType<ModuleOp>() : ModuleOp();
}


// Builds the symbol name for an outlined helper from its prefix and
// numeric id.

static std::string buildOutlinedFunctionName(StringRef prefix, int64_t id) {
  return (prefix + std::to_string(id)).str();
}


// Returns whether the function is an outlined helper that should not be
// processed again by the isolation pass.
static bool isOutlinedHelperFunction(func::FuncOp func) {
  if (!func)
    return false;

  StringRef name = func.getSymName();
  return name.starts_with(kMatrixInitializationPrefix) ||
         name.starts_with(kLinearRoutinePrefix) ||
         name.starts_with(kConv2DRoutinePrefix);
}


// Clones non-terminator operations from the source block into the
// builder while keeping the IR mapping current.

static void cloneOutlinedOpsIntoBuilder(Block &source, OpBuilder &builder,
                                        IRMapping &mapper) {
  for (Operation &op : source) {
    if (isa<cf::BranchOp, scf::YieldOp>(op))
      continue;
    Operation *cloned = builder.clone(op, mapper);
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
      mapper.map(oldRes, newRes);
  }
}


// Maps the exit-branch operands from the outlined layer body into the
// cloned function and returns them as function results.

static FailureOr<SmallVector<Value>> getOutlinedLayerReturns(Block &bodyBlk,
                                                             Block &exitBlk,
                                                             IRMapping &mapper) {
  auto br = dyn_cast<cf::BranchOp>(bodyBlk.getTerminator());
  if (!br || br.getDest() != &exitBlk)
    return failure();

  SmallVector<Value> returns;
  returns.reserve(br.getNumOperands());
  for (Value v : br.getDestOperands())
    returns.push_back(mapper.lookupOrDefault(v));
  return returns;
}


// Moves or clones constants into execute regions so outlined helpers
// do not depend on outer-scope constant definitions.

static void pullExternalConstantsIntoExecuteRegions(func::FuncOp func) {
  SmallVector<scf::ExecuteRegionOp> execs;
  func.walk([&](scf::ExecuteRegionOp exec) { execs.push_back(exec); });

  for (scf::ExecuteRegionOp exec : execs) {
    if (exec.getRegion().empty() || exec.getRegion().front().empty())
      continue;

    SmallVector<arith::ConstantOp> externalConsts = collectExternalConstantOpsForRegion(exec);
    if (externalConsts.empty())
      continue;

    Block &entry = exec.getRegion().front();
    OpBuilder b(exec.getContext());
    b.setInsertionPointToStart(&entry);

    for (arith::ConstantOp cst : externalConsts) {
      if (!cst || !cst->getBlock())
        continue;

      // Dense resource constants can be moved wholesale when all uses are
      // internal; everything else is cloned to avoid mutating outer uses.
      bool isDense = isa<DenseResourceElementsAttr>(cst.getValue());
      if (isDense && allUsesInsideRegion(cst, exec.getRegion())) {
        cst->moveBefore(&entry, entry.begin());
        continue;
      }

      auto cloned = cast<arith::ConstantOp>(b.clone(*cst.getOperation()));
      replaceUsesInsideRegion(cst.getResult(), cloned.getResult(), exec.getRegion());
    }
  }
}


// Moves top-level setup ops into execute regions when they are only used by
// the region, so outlined helpers do not gain avoidable call operands.
static void pullExternalProducersIntoExecuteRegions(func::FuncOp func) {
  SmallVector<scf::ExecuteRegionOp> execs;
  func.walk([&](scf::ExecuteRegionOp exec) { execs.push_back(exec); });

  for (scf::ExecuteRegionOp exec : execs) {
    if (exec.getRegion().empty())
      continue;

    Block &entry = exec.getRegion().front();
    bool changed = false;
    do {
      changed = false;

      SmallVector<Value> externalValues;
      collectExternalValuesForRegion(exec, externalValues);
      for (Value value : externalValues) {
        Operation *producer = value.getDefiningOp();
        if (!producer || isa<arith::ConstantOp>(producer))
          continue;
        if (!producer->getBlock() || producer->getBlock() != exec->getBlock())
          continue;
        if (findTopLevelOwner(producer) != producer)
          continue;
        if (!isAbsorbableTopLevelSetupOp(producer))
          continue;
        if (allUsesInsideRegion(producer, exec.getRegion())) {
          producer->moveBefore(&entry, entry.begin());
          changed = true;
          continue;
        }

        OpBuilder b(exec.getContext());
        b.setInsertionPointToStart(&entry);
        Operation *cloned = b.clone(*producer);
        for (auto [oldResult, newResult] :
             llvm::zip(producer->getResults(), cloned->getResults())) {
          replaceUsesInsideRegion(oldResult, newResult, exec.getRegion());
        }
        changed = true;
      }
    } while (changed);
  }
}


// Wraps matrix-initialization segments in execute regions so they can
// be outlined into helper functions later.

static void wrapMatrixInitializationChains(func::FuncOp func, OpChainList &matrixChains) {
  int64_t matrixInitCount = 0;
  for (auto &segment : matrixChains) {
    if (segment.empty())
      continue;
    Operation *first = segment.front();
    if (!first || !first->getBlock())
      continue;
    if (hasEscapingUses(segment))
      continue;

    OpBuilder b(first);
    // Execute regions give the later outlining step a single container to
    // replace with a helper function call.
    auto exec = b.create<scf::ExecuteRegionOp>(first->getLoc(), TypeRange{});
    exec->setAttr(kMatrixInitializationAttr,
                  IntegerAttr::get(IndexType::get(func.getContext()),
                                   matrixInitCount++));

    auto [body, exit] = createExecuteRegionBlocks(exec);
    moveSegmentIntoBlock(segment, body);

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
    bodyBuilder.create<cf::BranchOp>(first->getLoc(), exit);

    OpBuilder exitBuilder = OpBuilder::atBlockEnd(exit);
    exitBuilder.create<scf::YieldOp>(first->getLoc());
  }
}


// Wraps layer-routine segments in execute regions and returns any
// values that escape those segments.

static void wrapLayerRoutineChains(func::FuncOp func,
                                   TaggedOpChainList &layerRoutineChains) {
  int64_t layerRoutineCount = 0;
  for (auto &[segment, tag] : layerRoutineChains) {
    if (segment.empty())
      continue;
    Operation *first = segment.front();
    if (!first || !first->getBlock())
      continue;

    SmallVector<Value> escapingResults;
    computeEscapingResults(segment, escapingResults);

    SmallVector<Type> resultTypes;
    resultTypes.reserve(escapingResults.size());
    for (Value v : escapingResults)
      resultTypes.push_back(v.getType());

    OpBuilder b(first);
    auto exec = b.create<scf::ExecuteRegionOp>(first->getLoc(), resultTypes);
    exec->setAttr(tag,
                  IntegerAttr::get(IndexType::get(func.getContext()),
                                   layerRoutineCount++));

    auto [body, exit] = createExecuteRegionBlocks(exec);
    for (Value v : escapingResults)
      exit->addArgument(v.getType(), first->getLoc());

    DenseSet<Operation *> inChain = collectSegmentClosure(segment);
    moveSegmentIntoBlock(segment, body);

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
    bodyBuilder.create<cf::BranchOp>(first->getLoc(), exit, escapingResults);

    OpBuilder exitBuilder = OpBuilder::atBlockEnd(exit);
    exitBuilder.create<scf::YieldOp>(first->getLoc(), exit->getArguments());

    for (auto [oldValue, newValue] : llvm::zip(escapingResults, exec.getResults())) {
      oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        if (owner->getParentRegion() == &exec.getRegion())
          return false;
        return !inChain.contains(owner);
      });
    }
  }
}


// Outlines one matrix-initialization execute region into a private
// helper function and replaces the region with a call.

static void convertMatrixRegionToFunctionBody(func::FuncOp forward,
                                              scf::ExecuteRegionOp exec) {
  std::optional<int64_t> maybeId =
      getTaggedRegionId(exec, kMatrixInitializationAttr);
  if (!maybeId)
    return;
  if (exec.getNumResults() != 0)
    return;
  if (exec.getRegion().empty())
    return;

  ModuleOp module = getOutliningModule(forward);
  if (!module)
    return;

  int64_t id = *maybeId;
  std::string fnName =
      buildOutlinedFunctionName(kMatrixInitializationPrefix, id);

  func::FuncOp outlined = module.lookupSymbol<func::FuncOp>(fnName);
  if (!outlined) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());
    auto fnType = moduleBuilder.getFunctionType(TypeRange{}, TypeRange{});
    outlined = moduleBuilder.create<func::FuncOp>(exec.getLoc(), fnName, fnType);
    outlined.setPrivate();

    Block *entry = outlined.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    IRMapping mapper;

    // Matrix initialization execute regions are currently emitted in a
    // canonical 2-block shape: body block + exit block with scf.yield.
    for (Block &blk : exec.getRegion()) {
      cloneOutlinedOpsIntoBuilder(blk, b, mapper);
    }
    b.create<func::ReturnOp>(exec.getLoc());
  }

  OpBuilder b(exec);
  auto call = b.create<func::CallOp>(exec.getLoc(), outlined.getSymName(), TypeRange{}, ValueRange{});
  call->setAttr(kWeightIdAttr, b.getI64IntegerAttr(id));
  exec.erase();
}


// Converts all tagged matrix-initialization execute regions in the
// function into helper function calls.

static void convertMatrixRegionsToFunctionBodies(func::FuncOp forward) {
  SmallVector<scf::ExecuteRegionOp> execs =
      collectTaggedExecuteRegions(forward, kMatrixInitializationAttr);

  for (scf::ExecuteRegionOp exec : execs) {
    if (exec && exec->getBlock())
      convertMatrixRegionToFunctionBody(forward, exec);
  }
}


// Outlines one layer-routine execute region into a private helper
// function that preserves the original tensor input contract.

static void convertLayerRegionToFunctionBody(func::FuncOp forward,
                                             scf::ExecuteRegionOp exec,
                                             StringRef tag,
                                             StringRef prefix) {
  std::optional<int64_t> maybeId = getTaggedRegionId(exec, tag);
  if (!maybeId)
    return;
  if (exec.getRegion().empty() || exec.getRegion().getBlocks().size() < 2)
    return;

  ModuleOp module = getOutliningModule(forward);
  if (!module)
    return;

  int64_t id = *maybeId;
  std::string fnName = buildOutlinedFunctionName(prefix, id);
  SmallVector<Value> externalInputs;
  collectExternalValuesForRegion(exec, externalInputs);

  func::FuncOp outlined = module.lookupSymbol<func::FuncOp>(fnName);
  if (!outlined) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

    SmallVector<Type> argTypes;
    argTypes.reserve(externalInputs.size());
    for (Value input : externalInputs)
      argTypes.push_back(input.getType());
    SmallVector<Type> resTypes(exec.getResultTypes().begin(), exec.getResultTypes().end());
    auto fnType = moduleBuilder.getFunctionType(argTypes, resTypes);
    outlined = moduleBuilder.create<func::FuncOp>(exec.getLoc(), fnName, fnType);
    outlined.setPrivate();

    Block *entry = outlined.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    IRMapping mapper;
    for (auto [input, arg] : llvm::zip(externalInputs, entry->getArguments()))
      mapper.map(input, arg);

    Block &bodyBlk = exec.getRegion().front();
    Block &exitBlk = exec.getRegion().back();

    cloneOutlinedOpsIntoBuilder(bodyBlk, b, mapper);

    FailureOr<SmallVector<Value>> returns =
        getOutlinedLayerReturns(bodyBlk, exitBlk, mapper);
    if (failed(returns))
      return;
    b.create<func::ReturnOp>(exec.getLoc(), *returns);
  }

  OpBuilder b(exec);
  auto call = b.create<func::CallOp>(exec.getLoc(), outlined.getSymName(),
                                     exec.getResultTypes(), externalInputs);
  call->setAttr(kLayerIdAttr, b.getI64IntegerAttr(id));
  for (auto [oldResult, newResult] : llvm::zip(exec.getResults(), call.getResults())) {
    oldResult.replaceUsesWithIf(newResult, [&](OpOperand &use) {
      return use.getOwner() != call.getOperation();
    });
  }
  exec.erase();
}


// Converts all tagged layer-routine execute regions in the function
// into helper function calls.

static void convertLayerRegionsToFunctionBodies(func::FuncOp forward) {
  for (auto [tag, prefix] : {
           std::pair<StringRef, StringRef>{kLinearRoutineAttr,
                                           kLinearRoutinePrefix},
           std::pair<StringRef, StringRef>{kConv2DRoutineAttr,
                                           kConv2DRoutinePrefix},
       }) {
    SmallVector<scf::ExecuteRegionOp> execs =
        collectTaggedExecuteRegions(forward, tag);

    for (scf::ExecuteRegionOp exec : execs) {
      if (exec && exec->getBlock())
        convertLayerRegionToFunctionBody(forward, exec, tag, prefix);
    }
  }
}

} // namespace


// Exposes the CLI name used to invoke this pass from pass pipelines
// and tooling.

llvm::StringRef IsolateLayersPass::getArgument() const {
  return "analog-isolate-layers-and-weights";
}


// Summarizes the pass behavior for MLIR pass listings and debugging
// output.

llvm::StringRef IsolateLayersPass::getDescription() const {
  return "Isolate layer routines and weight-initialization routines into helper functions";
}


// Declares the dialects this pass may create while wrapping and
// outlining execute regions.

void IsolateLayersPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
}


// Finds analog layer and weight routines, wraps them in execute
// regions, and outlines them into helper functions.

void IsolateLayersPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (isOutlinedHelperFunction(func))
    return;

  OpChainList matrixChains;
  TaggedOpChainList layerRoutineChains;

  collectMatrixInitializationChains(func, matrixChains);
  collectLayerRoutineChains(func, layerRoutineChains);

  wrapMatrixInitializationChains(func, matrixChains);
  wrapLayerRoutineChains(func, layerRoutineChains);
  pullExternalProducersIntoExecuteRegions(func);
  pullExternalConstantsIntoExecuteRegions(func);
  convertMatrixRegionsToFunctionBodies(func);
  convertLayerRegionsToFunctionBodies(func);
}


// Builds a new instance of the pass for registration and pipeline
// construction.

std::unique_ptr<mlir::Pass> createIsolateLayersPass() {
  return std::make_unique<IsolateLayersPass>();
}

} // namespace analog
} // namespace mlir
