#include "analog-mlir/Dialect/Analog/Transforms/assemblers/TaskGraphAssemblyStep.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "analog-mlir/Dialect/Analog/Transforms/assemblers/TaskGraphAssemblyUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>

namespace {

inline constexpr llvm::StringLiteral kTaskGraphResourceCountAttrName(
    "analog.runtime.resource_count");
inline constexpr llvm::StringLiteral kTaskGraphInputSlotsAttrName(
    "analog.runtime.input_slots");
inline constexpr llvm::StringLiteral kTaskGraphOutputSlotsAttrName(
    "analog.runtime.output_slots");
inline constexpr llvm::StringLiteral kTaskGraphTempOffsetsAttrName(
    "analog.runtime.temp_offsets");
inline constexpr llvm::StringLiteral kTaskGraphTempBaseSlotAttrName(
    "analog.runtime.temp_base_slot");
inline constexpr llvm::StringLiteral kTaskGraphTempCountAttrName(
    "analog.runtime.temp_count");
inline constexpr llvm::StringLiteral kTaskGraphInitTaskCountAttrName(
    "analog.runtime.init_task_count");
inline constexpr llvm::StringLiteral kTaskGraphWorkspaceSizeAttrName(
    "analog.runtime.workspace_size");
inline constexpr llvm::StringLiteral kResourceSlotAttrName("analog.runtime.slot");
inline constexpr llvm::StringLiteral kResourceByteSizeAttrName(
    "analog.runtime.byte_size");
inline constexpr llvm::StringLiteral kResourceTempIndexAttrName(
    "analog.runtime.temp_index");
inline constexpr llvm::StringLiteral kResourceTempOffsetAttrName(
    "analog.runtime.temp_offset");
inline constexpr llvm::StringLiteral kTaskIndexAttrName(
    "analog.runtime.task_index");
inline constexpr llvm::StringLiteral kTaskPhaseAttrName(
    "analog.runtime.phase");
inline constexpr llvm::StringLiteral kTaskInputSlotsAttrName(
    "analog.runtime.input_slots");
inline constexpr llvm::StringLiteral kTaskOutputSlotsAttrName(
    "analog.runtime.output_slots");

constexpr size_t kWorkspaceAlignment = alignof(std::max_align_t);

struct TaskPlan {
  llvm::SmallVector<uint32_t, 4> inputSlots;
  llvm::SmallVector<uint32_t, 4> outputSlots;
  bool isInitTask = false;
};

struct ExecutablePlan {
  llvm::SmallVector<TaskPlan, 8> tasks;
  llvm::SmallVector<uint32_t, 4> inputSlots;
  llvm::SmallVector<uint32_t, 4> outputSlots;
  llvm::SmallVector<size_t, 4> tempOffsets;
  uint32_t resourceCount = 0;
  uint32_t initTaskCount = 0;
  uint32_t tempBaseSlot = 0;
  uint32_t tempCount = 0;
  size_t workspaceSize = 0;
};

struct ResourceInfo {
  uint32_t slot = 0;
  size_t byteSize = 0;
  std::optional<uint32_t> tempIndex;
};

struct TempInterval {
  uint32_t slot = 0;
  uint32_t tempIndex = 0;
  size_t byteSize = 0;
  unsigned firstUse = std::numeric_limits<unsigned>::max();
  unsigned lastUse = 0;
  size_t offset = 0;
};

struct ActiveAllocation {
  unsigned lastUse = 0;
  size_t offset = 0;
  size_t size = 0;
};

struct FreeAllocation {
  size_t offset = 0;
  size_t size = 0;
};

bool isInitTask(mlir::analog::TaskCreateOp taskOp) {
  auto layerType = taskOp.getLayerTypeAttr();
  return layerType && layerType.getValue() == "weight_init";
}

mlir::FailureOr<size_t> getStaticByteSize(mlir::Type valueType) {
  if (llvm::isa<mlir::analog::RuntimeHandleType>(valueType))
    return static_cast<size_t>(0);

  auto getByteSize =
      [](mlir::ShapedType shapedType) -> mlir::FailureOr<size_t> {
    if (!shapedType.hasStaticShape() || !shapedType.getElementType().isF32())
      return mlir::failure();

    int64_t elementCount = shapedType.getNumElements();
    if (elementCount < 0)
      return mlir::failure();

    return static_cast<size_t>(elementCount) * sizeof(float);
  };

  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(valueType)) {
    unsigned bitWidth = floatType.getWidth();
    if (bitWidth == 0)
      return mlir::failure();

    return llvm::divideCeil(static_cast<size_t>(bitWidth),
                            static_cast<size_t>(CHAR_BIT));
  }

  if (auto rankedTensorType = llvm::dyn_cast<mlir::RankedTensorType>(valueType))
    return getByteSize(rankedTensorType);

  if (auto memRefType = llvm::dyn_cast<mlir::MemRefType>(valueType))
    return getByteSize(memRefType);

  return mlir::failure();
}

mlir::FailureOr<size_t> getResourceByteSize(mlir::Value resourceValue) {
  auto resourceType =
      llvm::dyn_cast<mlir::analog::TaskResourceType>(resourceValue.getType());
  if (!resourceType)
    return mlir::failure();

  return getStaticByteSize(resourceType.getValueType());
}

template <typename OpT>
mlir::LogicalResult
recordResource(OpT resourceOp, ExecutablePlan &plan,
               llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
               llvm::SmallVectorImpl<mlir::Value> &temporaryResources) {
  mlir::FailureOr<size_t> byteSize = getResourceByteSize(resourceOp.getResult());
  if (failed(byteSize)) {
    resourceOp.emitError("expected runtime resources to carry runtime handles, "
                         "float scalars, or static f32 tensor/memref payloads");
    return mlir::failure();
  }

  ResourceInfo info;
  info.slot = resourceInfoByValue.size();
  info.byteSize = *byteSize;
  if constexpr (std::is_same_v<OpT, mlir::analog::TaskGraphTemporaryOp>) {
    info.tempIndex = temporaryResources.size();
    temporaryResources.push_back(resourceOp.getResult());
  }

  resourceInfoByValue.try_emplace(resourceOp.getResult(), info);

  if constexpr (std::is_same_v<OpT, mlir::analog::TaskGraphInputOp>) {
    plan.inputSlots.push_back(info.slot);
  } else if constexpr (std::is_same_v<OpT, mlir::analog::TaskGraphOutputOp>) {
    plan.outputSlots.push_back(info.slot);
  }

  return mlir::success();
}

mlir::LogicalResult
collectResources(mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
                 llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue,
                 llvm::SmallVectorImpl<mlir::Value> &temporaryResources) {
  mlir::Block &block = taskGraphFunc.getBody().front();
  for (mlir::Operation &op : block) {
    if (auto inputOp = llvm::dyn_cast<mlir::analog::TaskGraphInputOp>(&op)) {
      if (failed(recordResource(inputOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
      continue;
    }

    if (auto outputOp = llvm::dyn_cast<mlir::analog::TaskGraphOutputOp>(&op)) {
      if (failed(recordResource(outputOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
      continue;
    }

    if (auto temporaryOp =
            llvm::dyn_cast<mlir::analog::TaskGraphTemporaryOp>(&op)) {
      if (failed(recordResource(temporaryOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
      continue;
    }

    if (auto persistentOp =
            llvm::dyn_cast<mlir::analog::TaskGraphPersistentOp>(&op)) {
      if (failed(recordResource(persistentOp, plan, resourceInfoByValue,
                                temporaryResources)))
        return mlir::failure();
    }
  }

  plan.resourceCount = resourceInfoByValue.size();
  plan.tempCount = temporaryResources.size();
  plan.tempBaseSlot = plan.resourceCount;
  if (!temporaryResources.empty())
    plan.tempBaseSlot =
        resourceInfoByValue.lookup(temporaryResources.front()).slot;

  return mlir::success();
}

mlir::LogicalResult
collectTasks(mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
             llvm::DenseMap<mlir::Value, ResourceInfo> &resourceInfoByValue) {
  mlir::Block &block = taskGraphFunc.getBody().front();
  llvm::DenseMap<mlir::Value, unsigned> taskIndexByValue;
  bool sawRunTask = false;

  for (mlir::Operation &op : block) {
    auto taskOp = llvm::dyn_cast<mlir::analog::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    TaskPlan taskPlan;
    taskPlan.isInitTask = isInitTask(taskOp);

    if (!taskPlan.isInitTask)
      sawRunTask = true;
    else if (sawRunTask) {
      taskOp.emitError("expected init tasks to be frontloaded before run tasks");
      return mlir::failure();
    }

    for (mlir::Value dependency : taskOp.getDependencies()) {
      auto dependencyIt = taskIndexByValue.find(dependency);
      if (dependencyIt == taskIndexByValue.end() ||
          dependencyIt->second >= plan.tasks.size()) {
        taskOp.emitError("expected dependencies to reference earlier tasks");
        return mlir::failure();
      }
    }

    for (mlir::Value input : taskOp.getInputs()) {
      auto resourceIt = resourceInfoByValue.find(input);
      if (resourceIt == resourceInfoByValue.end()) {
        taskOp.emitError("expected every task input to have a runtime slot");
        return mlir::failure();
      }

      taskPlan.inputSlots.push_back(resourceIt->second.slot);
    }

    for (mlir::Value output : taskOp.getOutputs()) {
      auto resourceIt = resourceInfoByValue.find(output);
      if (resourceIt == resourceInfoByValue.end()) {
        taskOp.emitError("expected every task output to have a runtime slot");
        return mlir::failure();
      }

      taskPlan.outputSlots.push_back(resourceIt->second.slot);
    }

    if (taskPlan.isInitTask)
      ++plan.initTaskCount;

    taskIndexByValue.try_emplace(taskOp.getResult(), plan.tasks.size());
    plan.tasks.push_back(std::move(taskPlan));
  }

  return mlir::success();
}

void updateIntervalUse(TempInterval &interval, unsigned taskIndex) {
  interval.firstUse = std::min(interval.firstUse, taskIndex);
  interval.lastUse = std::max(interval.lastUse, taskIndex);
}

mlir::LogicalResult
packTemporaryWorkspace(mlir::func::FuncOp taskGraphFunc, ExecutablePlan &plan,
                       llvm::DenseMap<mlir::Value, ResourceInfo>
                           &resourceInfoByValue,
                       llvm::ArrayRef<mlir::Value> temporaryResources) {
  if (temporaryResources.empty()) {
    plan.workspaceSize = 0;
    plan.tempOffsets.clear();
    return mlir::success();
  }

  llvm::DenseMap<uint32_t, uint32_t> tempIndexBySlot;
  llvm::SmallVector<TempInterval> intervals;
  intervals.reserve(temporaryResources.size());
  for (mlir::Value temporaryResource : temporaryResources) {
    const ResourceInfo &resourceInfo =
        resourceInfoByValue.lookup(temporaryResource);
    TempInterval interval;
    interval.slot = resourceInfo.slot;
    interval.tempIndex = *resourceInfo.tempIndex;
    interval.byteSize = resourceInfo.byteSize;
    tempIndexBySlot.try_emplace(interval.slot, interval.tempIndex);
    intervals.push_back(interval);
  }

  for (const auto &task : llvm::enumerate(plan.tasks)) {
    auto recordSlots = [&](llvm::ArrayRef<uint32_t> slots) {
      for (uint32_t slot : slots) {
        auto tempIt = tempIndexBySlot.find(slot);
        if (tempIt == tempIndexBySlot.end())
          continue;

        updateIntervalUse(intervals[tempIt->second], task.index());
      }
    };

    recordSlots(task.value().inputSlots);
    recordSlots(task.value().outputSlots);
  }

  for (const TempInterval &interval : intervals) {
    if (interval.firstUse == std::numeric_limits<unsigned>::max()) {
      taskGraphFunc.emitError("expected every temporary slot to be used by at "
                              "least one task");
      return mlir::failure();
    }
  }

  std::sort(intervals.begin(), intervals.end(),
            [](const TempInterval &lhs, const TempInterval &rhs) {
              if (lhs.firstUse != rhs.firstUse)
                return lhs.firstUse < rhs.firstUse;
              return lhs.tempIndex < rhs.tempIndex;
            });

  llvm::SmallVector<ActiveAllocation> activeAllocations;
  llvm::SmallVector<FreeAllocation> freeAllocations;
  plan.tempOffsets.assign(temporaryResources.size(), 0);
  plan.workspaceSize = 0;

  for (TempInterval &interval : intervals) {
    llvm::erase_if(activeAllocations, [&](const ActiveAllocation &allocation) {
      if (allocation.lastUse >= interval.firstUse)
        return false;

      freeAllocations.push_back(
          FreeAllocation{allocation.offset, allocation.size});
      return true;
    });

    size_t chosenOffset = 0;
    auto reusableIt =
        std::find_if(freeAllocations.begin(), freeAllocations.end(),
                     [&](const FreeAllocation &allocation) {
                       return allocation.size >= interval.byteSize;
                     });

    if (reusableIt != freeAllocations.end()) {
      chosenOffset = reusableIt->offset;
      freeAllocations.erase(reusableIt);
    } else {
      chosenOffset = llvm::alignTo(plan.workspaceSize, kWorkspaceAlignment);
    }

    interval.offset = chosenOffset;
    plan.tempOffsets[interval.tempIndex] = chosenOffset;
    plan.workspaceSize =
        std::max(plan.workspaceSize, chosenOffset + interval.byteSize);
    activeAllocations.push_back(
        ActiveAllocation{interval.lastUse, chosenOffset, interval.byteSize});
  }

  return mlir::success();
}

mlir::ArrayAttr buildI64ArrayAttr(mlir::Builder &builder,
                                  llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

template <typename T>
mlir::ArrayAttr buildIntegerArrayAttr(mlir::Builder &builder,
                                      llvm::ArrayRef<T> values) {
  llvm::SmallVector<int64_t> widenedValues;
  widenedValues.reserve(values.size());
  for (T value : values)
    widenedValues.push_back(static_cast<int64_t>(value));
  return buildI64ArrayAttr(builder, widenedValues);
}

mlir::FailureOr<ExecutablePlan>
buildExecutablePlan(mlir::func::FuncOp taskGraphFunc) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected runtime-lowered task graph to have a "
                            "single block");
    return mlir::failure();
  }

  ExecutablePlan plan;
  llvm::DenseMap<mlir::Value, ResourceInfo> resourceInfoByValue;
  llvm::SmallVector<mlir::Value> temporaryResources;

  if (failed(collectResources(taskGraphFunc, plan, resourceInfoByValue,
                              temporaryResources)) ||
      failed(collectTasks(taskGraphFunc, plan, resourceInfoByValue)) ||
      failed(packTemporaryWorkspace(taskGraphFunc, plan, resourceInfoByValue,
                                    temporaryResources))) {
    return mlir::failure();
  }

  return plan;
}

mlir::LogicalResult annotateTaskGraphWithExecutablePlan(
    mlir::func::FuncOp taskGraphFunc, const ExecutablePlan &plan) {
  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected runtime-lowered task graph to have a "
                            "single block");
    return mlir::failure();
  }

  llvm::DenseMap<mlir::Value, ResourceInfo> resourceInfoByValue;
  llvm::SmallVector<mlir::Value> temporaryResources;
  ExecutablePlan recomputedPlan;
  if (failed(collectResources(taskGraphFunc, recomputedPlan, resourceInfoByValue,
                              temporaryResources))) {
    return mlir::failure();
  }

  mlir::Builder builder(taskGraphFunc.getContext());
  taskGraphFunc->setAttr(kTaskGraphResourceCountAttrName,
                         builder.getI64IntegerAttr(plan.resourceCount));
  taskGraphFunc->setAttr(kTaskGraphInputSlotsAttrName,
                         buildIntegerArrayAttr(
                             builder, llvm::ArrayRef<uint32_t>(plan.inputSlots)));
  taskGraphFunc->setAttr(
      kTaskGraphOutputSlotsAttrName,
      buildIntegerArrayAttr(builder,
                            llvm::ArrayRef<uint32_t>(plan.outputSlots)));
  taskGraphFunc->setAttr(kTaskGraphTempOffsetsAttrName,
                         buildIntegerArrayAttr(
                             builder, llvm::ArrayRef<size_t>(plan.tempOffsets)));
  taskGraphFunc->setAttr(kTaskGraphTempBaseSlotAttrName,
                         builder.getI64IntegerAttr(plan.tempBaseSlot));
  taskGraphFunc->setAttr(kTaskGraphTempCountAttrName,
                         builder.getI64IntegerAttr(plan.tempCount));
  taskGraphFunc->setAttr(kTaskGraphInitTaskCountAttrName,
                         builder.getI64IntegerAttr(plan.initTaskCount));
  taskGraphFunc->setAttr(kTaskGraphWorkspaceSizeAttrName,
                         builder.getI64IntegerAttr(plan.workspaceSize));

  for (mlir::Value temporaryResource : temporaryResources) {
    const ResourceInfo &resourceInfo =
        resourceInfoByValue.lookup(temporaryResource);
    mlir::Operation *resourceOp = temporaryResource.getDefiningOp();
    resourceOp->setAttr(kResourceTempIndexAttrName,
                        builder.getI64IntegerAttr(*resourceInfo.tempIndex));
    resourceOp->setAttr(
        kResourceTempOffsetAttrName,
        builder.getI64IntegerAttr(plan.tempOffsets[*resourceInfo.tempIndex]));
  }

  for (auto &resourceIt : resourceInfoByValue) {
    mlir::Operation *resourceOp = resourceIt.first.getDefiningOp();
    resourceOp->setAttr(kResourceSlotAttrName,
                        builder.getI64IntegerAttr(resourceIt.second.slot));
    resourceOp->setAttr(kResourceByteSizeAttrName,
                        builder.getI64IntegerAttr(resourceIt.second.byteSize));
  }

  unsigned taskIndex = 0;
  for (mlir::Operation &op : taskGraphFunc.getBody().front()) {
    auto taskOp = llvm::dyn_cast<mlir::analog::TaskCreateOp>(&op);
    if (!taskOp)
      continue;

    const TaskPlan &taskPlan = plan.tasks[taskIndex];
    taskOp->setAttr(kTaskIndexAttrName, builder.getI64IntegerAttr(taskIndex));
    taskOp->setAttr(kTaskPhaseAttrName,
                    builder.getStringAttr(taskPlan.isInitTask ? "init" : "run"));
    taskOp->setAttr(kTaskInputSlotsAttrName,
                    buildIntegerArrayAttr(
                        builder, llvm::ArrayRef<uint32_t>(taskPlan.inputSlots)));
    taskOp->setAttr(
        kTaskOutputSlotsAttrName,
        buildIntegerArrayAttr(builder,
                              llvm::ArrayRef<uint32_t>(taskPlan.outputSlots)));
    ++taskIndex;
  }

  return mlir::success();
}

class TaskGraphExecutionPlanAssembler final
    : public mlir::analog::TaskGraphAssemblyStep {
public:
  mlir::StringRef getName() const final { return "TaskGraphExecutionPlan"; }

  mlir::LogicalResult assemble(mlir::ModuleOp module,
                               mlir::func::FuncOp forward) const final {
    auto taskGraphFunc =
        mlir::analog::assembler_utils::lookupGeneratedTaskGraphFunc(module,
                                                                    forward);
    if (!taskGraphFunc) {
      forward.emitError("expected task graph scaffold to create a generator "
                        "function");
      return mlir::failure();
    }

    auto executablePlan = buildExecutablePlan(taskGraphFunc);
    if (failed(executablePlan) ||
        failed(annotateTaskGraphWithExecutablePlan(taskGraphFunc,
                                                   *executablePlan))) {
      return mlir::failure();
    }

    mlir::analog::assembler_utils::clearAssemblyAttrs(forward, taskGraphFunc);
    return mlir::success();
  }
};

} // namespace

namespace mlir {
namespace analog {

void registerTaskGraphExecutionPlanAssembler(TaskGraphAssemblySteps &steps) {
  steps.push_back(std::make_unique<TaskGraphExecutionPlanAssembler>());
}

} // namespace analog
} // namespace mlir
