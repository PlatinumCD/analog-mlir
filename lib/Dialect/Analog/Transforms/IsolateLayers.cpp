#include "analog-mlir/Dialect/Analog/Transforms/IsolateLayers.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

using OpChain = SmallVector<Operation *>;
using OpChainList = SmallVector<OpChain>;

constexpr StringLiteral kMatrixInitializationAttr = "analog-matrix-initialization";
constexpr StringLiteral kLayerRoutineAttr = "analog-layer-routine";
constexpr StringLiteral kWeightIdAttr = "weight-id";
constexpr StringLiteral kLayerIdAttr = "layer-id";
constexpr StringLiteral kMatrixInitializationPrefix = "analog_matrix_initialization_";
constexpr StringLiteral kLayerRoutinePrefix = "analog_layer_routine_";


// Walks up the parent chain until it finds the top-level operation
// directly owned by the surrounding function body.

static Operation *findTopLevelOwner(Operation *op) {
  Operation *top = op;
  while (top && !isa<func::FuncOp>(top->getParentOp()))
    top = top->getParentOp();
  return top;
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
                                      OpChainList &layerRoutineChains) {
  func.walk([&](analog::VectorFromTensorOp op) {
    Operation *startTop = findTopLevelOwner(op.getOperation());
    if (!startTop || !startTop->getBlock())
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
      layerRoutineChains.push_back(std::move(chain));
  });
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


// Checks whether all uses of the constant stay within the target
// region boundary.

static bool allUsesInsideRegion(arith::ConstantOp cst, Region &region) {
  for (Operation *user : cst->getUsers()) {
    if (!isOpInsideRegion(user, region))
      return false;
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


// Finds the first vector materialization in the execute region so the
// outlined layer keeps the original tensor input contract.

static analog::VectorFromTensorOp
findFirstVectorFromTensor(scf::ExecuteRegionOp exec) {
  analog::VectorFromTensorOp firstVectorFromTensor;
  exec.getRegion().walk([&](analog::VectorFromTensorOp op) {
    if (!firstVectorFromTensor)
      firstVectorFromTensor = op;
  });
  return firstVectorFromTensor;
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

static void wrapLayerRoutineChains(func::FuncOp func, OpChainList &layerRoutineChains) {
  int64_t layerRoutineCount = 0;
  for (auto &segment : layerRoutineChains) {
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
    exec->setAttr(kLayerRoutineAttr,
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
                                             scf::ExecuteRegionOp exec) {
  std::optional<int64_t> maybeId = getTaggedRegionId(exec, kLayerRoutineAttr);
  if (!maybeId)
    return;
  if (exec.getRegion().empty() || exec.getRegion().getBlocks().size() < 2)
    return;

  // The outlined layer entrypoint keeps the original tensor input contract:
  // whatever fed the first analog.vector.from_tensor becomes the function arg.
  analog::VectorFromTensorOp firstVectorFromTensor =
      findFirstVectorFromTensor(exec);
  if (!firstVectorFromTensor)
    return;
  Value layerInput = firstVectorFromTensor.getOperand();

  ModuleOp module = getOutliningModule(forward);
  if (!module)
    return;

  int64_t id = *maybeId;
  std::string fnName = buildOutlinedFunctionName(kLayerRoutinePrefix, id);

  func::FuncOp outlined = module.lookupSymbol<func::FuncOp>(fnName);
  if (!outlined) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

    SmallVector<Type> argTypes{layerInput.getType()};
    SmallVector<Type> resTypes(exec.getResultTypes().begin(), exec.getResultTypes().end());
    auto fnType = moduleBuilder.getFunctionType(argTypes, resTypes);
    outlined = moduleBuilder.create<func::FuncOp>(exec.getLoc(), fnName, fnType);
    outlined.setPrivate();

    Block *entry = outlined.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockEnd(entry);
    IRMapping mapper;
    mapper.map(layerInput, entry->getArgument(0));

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
                                     exec.getResultTypes(), ValueRange{layerInput});
  call->setAttr(kLayerIdAttr, b.getI64IntegerAttr(id));
  exec.replaceAllUsesWith(call.getResults());
  exec.erase();
}


// Converts all tagged layer-routine execute regions in the function
// into helper function calls.

static void convertLayerRegionsToFunctionBodies(func::FuncOp forward) {
  SmallVector<scf::ExecuteRegionOp> execs =
      collectTaggedExecuteRegions(forward, kLayerRoutineAttr);

  for (scf::ExecuteRegionOp exec : execs) {
    if (exec && exec->getBlock())
      convertLayerRegionToFunctionBody(forward, exec);
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
  OpChainList matrixChains;
  OpChainList layerRoutineChains;

  collectMatrixInitializationChains(func, matrixChains);
  collectLayerRoutineChains(func, layerRoutineChains);

  wrapMatrixInitializationChains(func, matrixChains);
  wrapLayerRoutineChains(func, layerRoutineChains);
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
