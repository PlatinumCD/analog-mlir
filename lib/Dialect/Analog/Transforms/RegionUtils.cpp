#include "analog-mlir/Dialect/Analog/Transforms/RegionUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace mlir {
namespace analog {
namespace detail {

Operation *findTopLevelOwner(Operation *op) {
  Operation *top = op;
  while (top && !isa<func::FuncOp>(top->getParentOp()))
    top = top->getParentOp();
  return top;
}

bool allUsesStayWithinTopLevelOwners(
    Operation *producer, const llvm::DenseSet<Operation *> &segmentOwners) {
  for (Value result : producer->getResults()) {
    for (Operation *user : result.getUsers()) {
      Operation *userTop = findTopLevelOwner(user);
      if (!userTop || !segmentOwners.contains(userTop))
        return false;
    }
  }
  return true;
}

llvm::DenseSet<Operation *> collectSegmentClosure(
    llvm::ArrayRef<Operation *> segment) {
  llvm::DenseSet<Operation *> inChain;
  for (Operation *cur : segment) {
    if (!cur)
      continue;
    inChain.insert(cur);
    cur->walk([&](Operation *nested) { inChain.insert(nested); });
  }
  return inChain;
}

bool isRegionInsideRegion(Region *candidate, Region &target) {
  for (Region *region = candidate; region;) {
    if (region == &target)
      return true;
    Operation *parent = region->getParentOp();
    region = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}

bool isOpInsideRegion(Operation *op, Region &region) {
  for (Region *r = op ? op->getParentRegion() : nullptr; r;) {
    if (r == &region)
      return true;
    Operation *parent = r->getParentOp();
    r = parent ? parent->getParentRegion() : nullptr;
  }
  return false;
}

bool isValueDefinedInsideRegion(Value value, Region &region) {
  if (!value)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    return owner && isRegionInsideRegion(owner->getParent(), region);
  }

  return isOpInsideRegion(value.getDefiningOp(), region);
}

void computeEscapingResults(llvm::ArrayRef<Operation *> segment,
                            llvm::SmallVectorImpl<Value> &escapingResults) {
  llvm::DenseSet<Operation *> inChain = collectSegmentClosure(segment);

  for (Operation *cur : segment) {
    if (!cur)
      continue;
    cur->walk([&](Operation *nested) {
      if (isa<tensor::EmptyOp>(nested))
        return;

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

void collectExternalValuesForRegion(scf::ExecuteRegionOp exec,
                                    llvm::SmallVectorImpl<Value> &externalValues) {
  llvm::DenseSet<Value> seen;
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

bool isValueAvailableAtOp(Value value, Operation *op) {
  if (!value || !op)
    return false;

  if (isa<BlockArgument>(value))
    return true;

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  if (definingOp->getBlock() != op->getBlock())
    return true;

  if (definingOp == op)
    return false;

  return definingOp->isBeforeInBlock(op);
}

void filterUndominatedExternalValues(scf::ExecuteRegionOp exec,
                                     llvm::SmallVectorImpl<Value> &values) {
  llvm::erase_if(values, [&](Value value) {
    return !isValueAvailableAtOp(value, exec.getOperation());
  });
}

bool allUsesInsideRegion(arith::ConstantOp cst, Region &region) {
  for (Operation *user : cst->getUsers()) {
    if (!isOpInsideRegion(user, region))
      return false;
  }
  return true;
}

bool allUsesInsideRegion(Operation *op, Region &region) {
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

void replaceUsesInsideRegion(Value oldValue, Value newValue, Region &region) {
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return isOpInsideRegion(use.getOwner(), region);
  });
}

void moveSegmentIntoBlock(llvm::ArrayRef<Operation *> segment, Block *body) {
  for (Operation *cur : segment) {
    if (!cur || !cur->getBlock())
      continue;
    cur->moveBefore(body, body->end());
  }
}

std::pair<Block *, Block *> createExecuteRegionBlocks(
    scf::ExecuteRegionOp exec) {
  Block *body = new Block();
  Block *exit = new Block();
  exec.getRegion().push_back(body);
  exec.getRegion().push_back(exit);
  return {body, exit};
}

} // namespace detail
} // namespace analog
} // namespace mlir
