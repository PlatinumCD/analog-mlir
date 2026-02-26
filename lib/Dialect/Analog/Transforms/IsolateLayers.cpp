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

using namespace mlir;

namespace mlir {
namespace analog {
namespace {

using OpChain = SmallVector<Operation *>;
using OpChainList = SmallVector<OpChain>;

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

    Operation *endTop = place;
    while (endTop && !isa<func::FuncOp>(endTop->getParentOp()))
      endTop = endTop->getParentOp();
    if (!endTop)
      return;
    if (cst->getBlock() != endTop->getBlock())
      return;

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

static void collectLayerRoutineChains(func::FuncOp func,
                                      OpChainList &layerRoutineChains) {
  func.walk([&](analog::VectorFromTensorOp op) {
    Operation *startTop = op.getOperation();
    while (startTop && !isa<func::FuncOp>(startTop->getParentOp()))
      startTop = startTop->getParentOp();
    if (!startTop || !startTop->getBlock())
      return;

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

static bool hasEscapingUses(ArrayRef<Operation *> segment) {
  DenseSet<Operation *> inChain;
  for (Operation *cur : segment) {
    if (!cur)
      continue;
    inChain.insert(cur);
    cur->walk([&](Operation *nested) { inChain.insert(nested); });
  }
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

static void computeEscapingResults(ArrayRef<Operation *> segment,
                                   SmallVectorImpl<Value> &escapingResults) {
  DenseSet<Operation *> inChain;
  for (Operation *cur : segment) {
    if (!cur)
      continue;
    inChain.insert(cur);
    cur->walk([&](Operation *nested) { inChain.insert(nested); });
  }

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

static bool isOpInsideRegion(Operation *op, Region &region) {
  for (Region *r = op ? op->getParentRegion() : nullptr; r; ) {
    if (r == &region)
      return true;
    Operation *parent = r->getParentOp();
    r = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}

static bool allUsesInsideRegion(arith::ConstantOp cst, Region &region) {
  for (Operation *user : cst->getUsers()) {
    if (!isOpInsideRegion(user, region))
      return false;
  }
  return true;
}

static void replaceUsesInsideRegion(Value oldValue, Value newValue, Region &region) {
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return isOpInsideRegion(use.getOwner(), region);
  });
}

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
    auto exec = b.create<scf::ExecuteRegionOp>(first->getLoc(), TypeRange{});
    exec->setAttr("analog-matrix-initialization",
                  IntegerAttr::get(IndexType::get(func.getContext()),
                                   matrixInitCount++));

    Block *body = new Block();
    Block *exit = new Block();
    exec.getRegion().push_back(body);
    exec.getRegion().push_back(exit);

    for (Operation *cur : segment) {
      if (!cur || !cur->getBlock())
        continue;
      cur->moveBefore(body, body->end());
    }

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
    bodyBuilder.create<cf::BranchOp>(first->getLoc(), exit);

    OpBuilder exitBuilder = OpBuilder::atBlockEnd(exit);
    exitBuilder.create<scf::YieldOp>(first->getLoc());
  }
}

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
    exec->setAttr("analog-layer-routine",
                  IntegerAttr::get(IndexType::get(func.getContext()),
                                   layerRoutineCount++));

    Block *body = new Block();
    Block *exit = new Block();
    for (Value v : escapingResults)
      exit->addArgument(v.getType(), first->getLoc());
    exec.getRegion().push_back(body);
    exec.getRegion().push_back(exit);

    DenseSet<Operation *> inChain;
    for (Operation *cur : segment) {
      if (!cur || !cur->getBlock())
        continue;
      inChain.insert(cur);
      cur->walk([&](Operation *nested) { inChain.insert(nested); });
      cur->moveBefore(body, body->end());
    }

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

static void convertMatrixRegionToFunctionBody(func::FuncOp forward,
                                              scf::ExecuteRegionOp exec) {
  auto attr = exec->getAttrOfType<IntegerAttr>("analog-matrix-initialization");
  if (!attr)
    return;
  if (exec.getNumResults() != 0)
    return;
  if (exec.getRegion().empty())
    return;

  ModuleOp module = forward->getParentOfType<ModuleOp>();
  if (!module)
    return;

  int64_t id = attr.getValue().getSExtValue();
  std::string fnName = "analog_matrix_initialization_" + std::to_string(id);

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
      for (Operation &op : blk) {
        if (isa<cf::BranchOp, scf::YieldOp>(op))
          continue;
        Operation *cloned = b.clone(op, mapper);
        for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
          mapper.map(oldRes, newRes);
      }
    }
    b.create<func::ReturnOp>(exec.getLoc());
  }

  OpBuilder b(exec);
  auto call = b.create<func::CallOp>(exec.getLoc(), outlined.getSymName(), TypeRange{}, ValueRange{});
  call->setAttr("weight-id", b.getI64IntegerAttr(id));
  exec.erase();
}

static void convertMatrixRegionsToFunctionBodies(func::FuncOp forward) {
  SmallVector<scf::ExecuteRegionOp> execs;
  forward.walk([&](scf::ExecuteRegionOp exec) {
    if (exec->hasAttr("analog-matrix-initialization"))
      execs.push_back(exec);
  });

  for (scf::ExecuteRegionOp exec : execs) {
    if (exec && exec->getBlock())
      convertMatrixRegionToFunctionBody(forward, exec);
  }
}

static void convertLayerRegionToFunctionBody(func::FuncOp forward,
                                             scf::ExecuteRegionOp exec) {
  auto attr = exec->getAttrOfType<IntegerAttr>("analog-layer-routine");
  if (!attr)
    return;
  if (exec.getRegion().empty() || exec.getRegion().getBlocks().size() < 2)
    return;

  // The layer routine function takes the tensor feeding the first
  // analog.vector.from_tensor in the region.
  analog::VectorFromTensorOp firstVectorFromTensor;
  exec.getRegion().walk([&](analog::VectorFromTensorOp op) {
    if (!firstVectorFromTensor)
      firstVectorFromTensor = op;
  });
  if (!firstVectorFromTensor)
    return;
  Value layerInput = firstVectorFromTensor.getOperand();

  ModuleOp module = forward->getParentOfType<ModuleOp>();
  if (!module)
    return;

  int64_t id = attr.getValue().getSExtValue();
  std::string fnName = "analog_layer_routine_" + std::to_string(id);

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

    for (Operation &op : bodyBlk) {
      if (isa<cf::BranchOp, scf::YieldOp>(op))
        continue;
      Operation *cloned = b.clone(op, mapper);
      for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
        mapper.map(oldRes, newRes);
    }

    auto br = dyn_cast<cf::BranchOp>(bodyBlk.getTerminator());
    if (!br || br.getDest() != &exitBlk)
      return;

    SmallVector<Value> returns;
    returns.reserve(br.getNumOperands());
    for (Value v : br.getDestOperands())
      returns.push_back(mapper.lookupOrDefault(v));
    b.create<func::ReturnOp>(exec.getLoc(), returns);
  }

  OpBuilder b(exec);
  auto call = b.create<func::CallOp>(exec.getLoc(), outlined.getSymName(),
                                     exec.getResultTypes(), ValueRange{layerInput});
  call->setAttr("layer-id", b.getI64IntegerAttr(id));
  exec.replaceAllUsesWith(call.getResults());
  exec.erase();
}

static void convertLayerRegionsToFunctionBodies(func::FuncOp forward) {
  SmallVector<scf::ExecuteRegionOp> execs;
  forward.walk([&](scf::ExecuteRegionOp exec) {
    if (exec->hasAttr("analog-layer-routine"))
      execs.push_back(exec);
  });

  for (scf::ExecuteRegionOp exec : execs) {
    if (exec && exec->getBlock())
      convertLayerRegionToFunctionBody(forward, exec);
  }
}

} // namespace

llvm::StringRef IsolateLayersPass::getArgument() const {
  return "analog-isolate-layers-and-weights";
}

llvm::StringRef IsolateLayersPass::getDescription() const {
  return "Isolate layer routines and weight-initialization routines into helper functions";
}

void IsolateLayersPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
}

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

std::unique_ptr<mlir::Pass> createIsolateLayersPass() {
  return std::make_unique<IsolateLayersPass>();
}

} // namespace analog
} // namespace mlir
