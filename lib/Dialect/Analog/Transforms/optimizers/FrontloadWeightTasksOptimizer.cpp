#include "analog-mlir/Dialect/Analog/Transforms/optimizers/FrontloadWeightTasksOptimizer.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

namespace {

// Keeps task-create operation lists in block order without owning the ops.
using TaskList = llvm::SmallVector<mlir::analog::TaskCreateOp>;

// Recognizes helper tasks emitted by weight isolation through the layer marker.
bool isWeightTask(mlir::analog::TaskCreateOp taskOp) {
  auto layerType = taskOp.getLayerTypeAttr();
  return layerType && layerType.getValue() == "weight_init";
}

// Extracts task nodes from a graph block while preserving their current order.
TaskList collectTaskCreateOps(mlir::Block &block) {
  TaskList tasks;
  for (mlir::Operation &op : block) {
    if (auto taskOp = llvm::dyn_cast<mlir::analog::TaskCreateOp>(&op))
      tasks.push_back(taskOp);
  }

  return tasks;
}

// Splits weight initialization tasks from all other work without reordering
// either group.
void partitionTasksByWeight(const TaskList &orderedTasks, TaskList &weightTasks,
                            TaskList &nonWeightTasks) {
  for (mlir::analog::TaskCreateOp taskOp : orderedTasks) {
    if (isWeightTask(taskOp))
      weightTasks.push_back(taskOp);
    else
      nonWeightTasks.push_back(taskOp);
  }
}

// Builds the canonical schedule: all weight setup first, then original work.
TaskList buildFrontloadedOrder(const TaskList &orderedTasks,
                               const TaskList &weightTasks,
                               const TaskList &nonWeightTasks) {
  TaskList desiredOrder;
  desiredOrder.reserve(orderedTasks.size());
  desiredOrder.append(weightTasks.begin(), weightTasks.end());
  desiredOrder.append(nonWeightTasks.begin(), nonWeightTasks.end());
  return desiredOrder;
}

// Moves existing task ops to the desired order at the first task position.
void applyTaskOrder(TaskList &orderedTasks, const TaskList &desiredOrder) {
  mlir::Operation *anchor = orderedTasks.front().getOperation();
  for (mlir::analog::TaskCreateOp taskOp : desiredOrder) {
    if (taskOp.getOperation() == anchor) {
      anchor = anchor->getNextNode();
      continue;
    }

    taskOp->moveBefore(anchor);
  }
}

// Schedules weight initialization before compute while preserving task objects.
class FrontloadWeightTasksOptimizer
    : public mlir::analog::SymbolTaskGraphOptimizer {
public:
  // Names this optimizer in task graph diagnostics.
  mlir::StringRef getName() const final { return "FrontloadWeightTasks"; }

  // Reorders a generated single-block task graph so weights are ready first.
  mlir::LogicalResult optimize(mlir::func::FuncOp taskGraphFunc) const final {
    if (!taskGraphFunc.getBody().hasOneBlock()) {
      taskGraphFunc.emitError(
          "expected task graph function to have a single block");
      return mlir::failure();
    }

    mlir::Block &block = taskGraphFunc.getBody().front();
    TaskList orderedTasks = collectTaskCreateOps(block);

    // Separate the current schedule into stable partitions before rebuilding.
    TaskList weightTasks;
    TaskList nonWeightTasks;
    partitionTasksByWeight(orderedTasks, weightTasks, nonWeightTasks);

    if (weightTasks.empty() || nonWeightTasks.empty())
      return mlir::success();

    // Materialize only when the stable frontloaded order differs from the IR.
    TaskList desiredOrder =
        buildFrontloadedOrder(orderedTasks, weightTasks, nonWeightTasks);
    if (llvm::equal(orderedTasks, desiredOrder))
      return mlir::success();

    applyTaskOrder(orderedTasks, desiredOrder);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace analog {

// Installs the weight-frontloading optimizer into the symbolic graph pipeline.
void registerFrontloadWeightTasksOptimizer(
    SymbolTaskGraphOptimizers &optimizers) {
  optimizers.push_back(std::make_unique<FrontloadWeightTasksOptimizer>());
}

} // namespace analog
} // namespace mlir
