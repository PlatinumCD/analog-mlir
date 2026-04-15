#include "analog-mlir/Dialect/Analog/Transforms/optimizers/CoreScheduleLinearOptimizer.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <memory>

namespace {

inline constexpr llvm::StringLiteral kTaskCoreIdAttrName(
    "analog.runtime.core_id");

// Keeps task-create operation lists in block order without owning the ops.
using TaskList = llvm::SmallVector<mlir::analog::TaskCreateOp>;

bool isWeightTask(mlir::analog::TaskCreateOp taskOp) {
  auto layerType = taskOp.getLayerTypeAttr();
  return layerType && layerType.getValue() == "weight_init";
}

bool isAnalogTask(mlir::analog::TaskCreateOp taskOp) {
  return taskOp.getDomain() == "analog";
}

TaskList collectTaskCreateOps(mlir::Block &block) {
  TaskList tasks;
  for (mlir::Operation &op : block) {
    if (auto taskOp = llvm::dyn_cast<mlir::analog::TaskCreateOp>(&op))
      tasks.push_back(taskOp);
  }

  return tasks;
}

mlir::LogicalResult failIfCoreIdsAlreadyPresent(llvm::ArrayRef<mlir::analog::TaskCreateOp> tasks) {
  for (mlir::analog::TaskCreateOp taskOp : tasks) {
    if (taskOp->hasAttr(kTaskCoreIdAttrName)) {
      taskOp.emitError("expected tasks to not carry '")
          << kTaskCoreIdAttrName << "' before CoreScheduleLinearOptimizer";
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult collectDirectWeightDependents(
    llvm::ArrayRef<mlir::analog::TaskCreateOp> tasks,
    llvm::DenseMap<mlir::Operation *, TaskList> &dependentsByWeightTask) {
  dependentsByWeightTask.clear();

  llvm::DenseMap<mlir::Value, mlir::analog::TaskCreateOp> taskByResult;
  for (mlir::analog::TaskCreateOp taskOp : tasks)
    taskByResult.try_emplace(taskOp.getResult(), taskOp);

  for (mlir::analog::TaskCreateOp taskOp : tasks) {
    TaskList weightDependencies;
    for (mlir::Value dependency : taskOp.getDependencies()) {
      auto dependencyIt = taskByResult.find(dependency);
      if (dependencyIt == taskByResult.end())
        continue;
      if (!isWeightTask(dependencyIt->second))
        continue;

      weightDependencies.push_back(dependencyIt->second);
    }

    if (weightDependencies.size() > 1) {
      taskOp.emitError("expected a task to directly depend on at most one "
                       "weight_init task");
      return mlir::failure();
    }

    if (weightDependencies.empty())
      continue;

    if (!isAnalogTask(taskOp)) {
      taskOp.emitError("expected direct dependents of weight_init tasks to "
                       "have domain \"analog\"");
      return mlir::failure();
    }

    dependentsByWeightTask[weightDependencies.front().getOperation()].push_back(
        taskOp);
  }

  return mlir::success();
}

mlir::LogicalResult assignCoreId(mlir::analog::TaskCreateOp taskOp,
                                 int64_t coreId,
                                 llvm::DenseMap<mlir::Operation *, int64_t>
                                     &assignedCoreIds) {
  auto assignedIt = assignedCoreIds.find(taskOp.getOperation());
  if (assignedIt != assignedCoreIds.end()) {
    if (assignedIt->second != coreId) {
      taskOp.emitError("expected each task to map to exactly one core id");
      return mlir::failure();
    }
    return mlir::success();
  }

  taskOp->setAttr(kTaskCoreIdAttrName,
                  mlir::IntegerAttr::get(mlir::IntegerType::get(
                                             taskOp.getContext(), 64),
                                         coreId));
  assignedCoreIds.try_emplace(taskOp.getOperation(), coreId);
  return mlir::success();
}

mlir::LogicalResult assignCoreIdsInWeightOrder(
    llvm::ArrayRef<mlir::analog::TaskCreateOp> tasks,
    llvm::DenseMap<mlir::Operation *, TaskList> &dependentsByWeightTask) {
  llvm::DenseMap<mlir::Operation *, int64_t> assignedCoreIds;
  int64_t nextCoreId = 1;

  for (mlir::analog::TaskCreateOp taskOp : tasks) {
    if (!isWeightTask(taskOp))
      continue;

    if (!isAnalogTask(taskOp)) {
      taskOp.emitError("expected weight_init tasks to have domain \"analog\"");
      return mlir::failure();
    }

    int64_t coreId = nextCoreId++;
    if (failed(assignCoreId(taskOp, coreId, assignedCoreIds)))
      return mlir::failure();

    auto dependentIt = dependentsByWeightTask.find(taskOp.getOperation());
    if (dependentIt == dependentsByWeightTask.end())
      continue;

    for (mlir::analog::TaskCreateOp dependentTask : dependentIt->second)
      if (failed(assignCoreId(dependentTask, coreId, assignedCoreIds)))
        return mlir::failure();
  }

  return mlir::success();
}

// Assigns 1-based core ids to weight-init tasks and their direct analog users.
class CoreScheduleLinearOptimizer
    : public mlir::analog::SymbolTaskGraphOptimizer {
public:
  // Names this optimizer in task graph diagnostics.
  mlir::StringRef getName() const final { return "CoreScheduleLinear"; }

  // Annotates weight-backed analog work with stable 1-based core ids.
  mlir::LogicalResult optimize(mlir::func::FuncOp taskGraphFunc) const final {
    if (!taskGraphFunc.getBody().hasOneBlock()) {
      taskGraphFunc.emitError(
          "expected task graph function to have a single block");
      return mlir::failure();
    }

    mlir::Block &block = taskGraphFunc.getBody().front();
    TaskList tasks = collectTaskCreateOps(block);

    if (failed(failIfCoreIdsAlreadyPresent(tasks)))
      return mlir::failure();

    llvm::DenseMap<mlir::Operation *, TaskList> dependentsByWeightTask;
    if (failed(collectDirectWeightDependents(tasks, dependentsByWeightTask)))
      return mlir::failure();

    return assignCoreIdsInWeightOrder(tasks, dependentsByWeightTask);
  }
};

} // namespace

namespace mlir {
namespace analog {

// Installs the core linear scheduling optimizer into the symbolic graph
// optimizer pipeline.
void registerCoreScheduleLinearOptimizer(
    SymbolTaskGraphOptimizers &optimizers) {
  optimizers.push_back(std::make_unique<CoreScheduleLinearOptimizer>());
}

} // namespace analog
} // namespace mlir
