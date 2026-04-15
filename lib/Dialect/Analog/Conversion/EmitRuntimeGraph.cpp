#include "analog-mlir/Dialect/Analog/Conversion/EmitRuntimeGraph.h"

#include "analog-mlir/Dialect/Analog/IR/AnalogOps.h"
#include "analog-mlir/Dialect/Analog/IR/AnalogTypes.h"

#include "TaskGraphRuntime.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir {
namespace analog {

namespace {

inline constexpr llvm::StringLiteral kTaskGraphWorkspaceSizeAttrName(
    "analog.runtime.workspace_size");
inline constexpr llvm::StringLiteral kTaskGraphInitTaskCountAttrName(
    "analog.runtime.init_task_count");
inline constexpr llvm::StringLiteral kResourceSlotAttrName("analog.runtime.slot");
inline constexpr llvm::StringLiteral kResourceByteSizeAttrName(
    "analog.runtime.byte_size");
inline constexpr llvm::StringLiteral kResourceTempOffsetAttrName(
    "analog.runtime.temp_offset");
inline constexpr llvm::StringLiteral kTaskIndexAttrName(
    "analog.runtime.task_index");
inline constexpr llvm::StringLiteral kTaskPhaseAttrName(
    "analog.runtime.phase");
inline constexpr llvm::StringLiteral kTaskCoreIdAttrName(
    "analog.runtime.core_id");

struct ResourceModel {
  Value value;
  Type valueType;
  int32_t kind = RES_BUFFER;
  int32_t storage = STORAGE_TEMP;
  uint32_t slot = 0;
  uint64_t byteSize = 0;
  uint64_t workspaceOffset = 0;
};

struct BindingModel {
  int32_t kind = ARG_BUFFER;
  uint16_t flags = 0;
  int32_t source = SRC_SLOT;
  uint32_t sourceIndex = 0;
  uint32_t byteOffset = 0;
  uint32_t byteSize = 0;
};

struct TaskModel {
  TaskCreateOp op;
  uint32_t callableIndex = 0;
  uint16_t phase = TASK_PHASE_RUN;
  int32_t coreId = -1;
  uint32_t argBegin = 0;
  uint16_t argCount = 0;
  uint32_t depBegin = 0;
  uint16_t depCount = 0;
  uint32_t payloadOffset = 0;
  uint32_t payloadSize = 0;
};

struct CallableModel {
  std::string symbol;
  LLVM::LLVMFuncOp callee;
  TaskCreateOp representativeTask;
};

struct GraphModel {
  SmallVector<ResourceModel> resources;
  SmallVector<CallableModel> callables;
  SmallVector<BindingModel> bindings;
  SmallVector<uint32_t> deps;
  SmallVector<TaskModel> tasks;
  uint32_t initTaskCount = 0;
  uint64_t workspaceSize = 0;
};

struct RuntimeDecls {
  LLVM::LLVMFuncOp graphCreate;
  LLVM::LLVMFuncOp graphSetResource;
  LLVM::LLVMFuncOp graphSetCallable;
  LLVM::LLVMFuncOp graphSetTask;
  LLVM::LLVMFuncOp graphSetBinding;
  LLVM::LLVMFuncOp graphSetDep;
  LLVM::LLVMFuncOp runtimeInit;
  LLVM::LLVMFuncOp runtimeExecute;
  LLVM::LLVMFuncOp runtimeDestroy;
  LLVM::LLVMFuncOp taskArgBuffer;
  LLVM::LLVMFuncOp taskSetArgHandle;
  LLVM::LLVMFuncOp copyToBuffer;
  LLVM::LLVMFuncOp persistentHandleCreate;
  LLVM::LLVMFuncOp freeFunc;
};

std::string sanitizeSymbolSuffix(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size());
  for (char c : value)
    result.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  return result;
}

std::string makeUniqueSymbolName(ModuleOp module, llvm::StringRef prefix,
                                 llvm::StringRef suffix) {
  std::string baseName = (prefix + sanitizeSymbolSuffix(suffix)).str();
  if (!module.lookupSymbol(baseName))
    return baseName;

  unsigned disambiguator = 0;
  std::string candidate = baseName + "_" + std::to_string(disambiguator);
  while (module.lookupSymbol(candidate)) {
    ++disambiguator;
    candidate = baseName + "_" + std::to_string(disambiguator);
  }
  return candidate;
}

void setPrivateVisibility(Operation *op, Builder &builder) {
  op->setAttr(SymbolTable::getVisibilityAttrName(),
              builder.getStringAttr("private"));
}

FailureOr<uint64_t> getRequiredI64Attr(Operation *op, llvm::StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    op->emitError("expected required runtime attr '") << name << "'";
    return failure();
  }
  if (attr.getInt() < 0) {
    op->emitError("expected non-negative runtime attr '") << name << "'";
    return failure();
  }
  return static_cast<uint64_t>(attr.getInt());
}

uint64_t lookupTaskIndexOrMax(TaskCreateOp taskOp) {
  auto taskIndex = getRequiredI64Attr(taskOp, kTaskIndexAttrName);
  return succeeded(taskIndex) ? *taskIndex
                              : std::numeric_limits<uint64_t>::max();
}

FailureOr<uint16_t> getTaskPhase(TaskCreateOp taskOp) {
  auto phaseAttr = taskOp->getAttrOfType<StringAttr>(kTaskPhaseAttrName);
  if (phaseAttr) {
    if (phaseAttr.getValue() == "init")
      return static_cast<uint16_t>(TASK_PHASE_INIT);
    if (phaseAttr.getValue() == "run")
      return static_cast<uint16_t>(TASK_PHASE_RUN);
  }

  auto layerType = taskOp.getLayerTypeAttr();
  if (layerType && layerType.getValue() == "weight_init")
    return static_cast<uint16_t>(TASK_PHASE_INIT);
  return static_cast<uint16_t>(TASK_PHASE_RUN);
}

FailureOr<int32_t> getTaskCoreId(TaskCreateOp taskOp) {
  auto coreIdAttr = taskOp->getAttrOfType<IntegerAttr>(kTaskCoreIdAttrName);
  if (!coreIdAttr)
    return static_cast<int32_t>(-1);

  int64_t coreId = coreIdAttr.getInt();
  if (coreId <= 0 || coreId > std::numeric_limits<int32_t>::max()) {
    taskOp.emitError("expected runtime attr '")
        << kTaskCoreIdAttrName << "' to be a positive 32-bit integer";
    return failure();
  }

  return static_cast<int32_t>(coreId);
}

bool isHandleResourceType(Type valueType) {
  return isa<RuntimeHandleType>(valueType);
}

FailureOr<ShapedType> getSupportedBufferType(Operation *op, Type valueType) {
  auto shapedType = dyn_cast<ShapedType>(valueType);
  if (!shapedType || !shapedType.hasStaticShape() ||
      !shapedType.getElementType().isF32()) {
    op->emitError("expected runtime shims to lower only static shaped f32 "
                  "buffer resources");
    return failure();
  }

  return shapedType;
}

SmallVector<int64_t> computeRowMajorStrides(ShapedType shapedType) {
  SmallVector<int64_t> strides(shapedType.getRank(), 1);
  int64_t stride = 1;
  for (int64_t index = shapedType.getRank() - 1; index >= 0; --index) {
    strides[index] = stride;
    stride *= shapedType.getDimSize(index);
  }
  return strides;
}

LLVM::LLVMStructType getBufferViewType(MLIRContext *context) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);
  return LLVM::LLVMStructType::getLiteral(context, {ptrType, i64Type});
}

LLVM::LLVMStructType getMemRefDescriptorType(MLIRContext *context,
                                             ShapedType shapedType) {
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i64Type = IntegerType::get(context, 64);
  unsigned rank = shapedType.getRank();
  auto indexArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
  return LLVM::LLVMStructType::getLiteral(
      context, {ptrType, ptrType, i64Type, indexArrayType, indexArrayType});
}

Value buildI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                          builder.getI32IntegerAttr(value));
}

Value buildI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                          builder.getI64IntegerAttr(value));
}

Value buildI16Constant(OpBuilder &builder, Location loc, int16_t value) {
  return builder.create<LLVM::ConstantOp>(loc, builder.getI16Type(),
                                          builder.getI16IntegerAttr(value));
}

Value buildZeroPointer(OpBuilder &builder, Location loc) {
  return builder.create<LLVM::ZeroOp>(loc, LLVM::LLVMPointerType::get(
                                               builder.getContext()));
}

Value buildMemRefDescriptor(OpBuilder &builder, Location loc,
                            ShapedType shapedType, Value dataPtr) {
  auto descriptorType = getMemRefDescriptorType(builder.getContext(), shapedType);
  Value descriptor = builder.create<LLVM::ZeroOp>(loc, descriptorType);
  descriptor =
      builder.create<LLVM::InsertValueOp>(loc, descriptor, dataPtr,
                                          ArrayRef<int64_t>{0});
  descriptor =
      builder.create<LLVM::InsertValueOp>(loc, descriptor, dataPtr,
                                          ArrayRef<int64_t>{1});
  descriptor = builder.create<LLVM::InsertValueOp>(
      loc, descriptor, buildI64Constant(builder, loc, 0), ArrayRef<int64_t>{2});

  SmallVector<int64_t> strides = computeRowMajorStrides(shapedType);
  for (auto indexedDim : llvm::enumerate(shapedType.getShape())) {
    descriptor = builder.create<LLVM::InsertValueOp>(
        loc, descriptor,
        buildI64Constant(builder, loc, indexedDim.value()),
        ArrayRef<int64_t>{3, static_cast<int64_t>(indexedDim.index())});
    descriptor = builder.create<LLVM::InsertValueOp>(
        loc, descriptor,
        buildI64Constant(builder, loc, strides[indexedDim.index()]),
        ArrayRef<int64_t>{4, static_cast<int64_t>(indexedDim.index())});
  }

  return descriptor;
}

void flattenMemRefDescriptor(OpBuilder &builder, Location loc, Value descriptor,
                             ShapedType shapedType,
                             SmallVectorImpl<Value> &operands) {
  operands.push_back(builder.create<LLVM::ExtractValueOp>(
      loc, descriptor, ArrayRef<int64_t>{0}));
  operands.push_back(builder.create<LLVM::ExtractValueOp>(
      loc, descriptor, ArrayRef<int64_t>{1}));
  operands.push_back(builder.create<LLVM::ExtractValueOp>(
      loc, descriptor, ArrayRef<int64_t>{2}));
  for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
    operands.push_back(builder.create<LLVM::ExtractValueOp>(
        loc, descriptor, ArrayRef<int64_t>{3, dim}));
  for (int64_t dim = 0; dim < shapedType.getRank(); ++dim)
    operands.push_back(builder.create<LLVM::ExtractValueOp>(
        loc, descriptor, ArrayRef<int64_t>{4, dim}));
}

LLVM::LLVMFuncOp getOrCreateExternFunc(ModuleOp module, StringRef name,
                                       LLVM::LLVMFunctionType type) {
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

RuntimeDecls getRuntimeDecls(ModuleOp module) {
  MLIRContext *context = module.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type voidType = LLVM::LLVMVoidType::get(context);
  Type i16Type = IntegerType::get(context, 16);
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  auto bufferViewType = getBufferViewType(context);

  RuntimeDecls decls;
  decls.graphCreate = getOrCreateExternFunc(
      module, "analog_runtime_graph_create",
      LLVM::LLVMFunctionType::get(
          ptrType,
          {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
           i32Type, i64Type},
          false));
  decls.graphSetResource = getOrCreateExternFunc(
      module, "analog_runtime_graph_set_resource",
      LLVM::LLVMFunctionType::get(
          voidType, {ptrType, i32Type, i32Type, i32Type, i32Type, i64Type,
                     i64Type},
          false));
  decls.graphSetCallable = getOrCreateExternFunc(
      module, "analog_runtime_graph_set_callable",
      LLVM::LLVMFunctionType::get(
          voidType, {ptrType, i32Type, i32Type, ptrType, i32Type}, false));
  decls.graphSetTask = getOrCreateExternFunc(
      module, "analog_runtime_graph_set_task",
      LLVM::LLVMFunctionType::get(
          voidType,
          {ptrType, i32Type, i32Type, i32Type, i16Type, i16Type, i32Type,
           i16Type, i32Type, i32Type, i32Type},
          false));
  decls.graphSetBinding = getOrCreateExternFunc(
      module, "analog_runtime_graph_set_binding",
      LLVM::LLVMFunctionType::get(
          voidType, {ptrType, i32Type, i32Type, i16Type, i32Type, i32Type,
                     i32Type, i32Type},
          false));
  decls.graphSetDep = getOrCreateExternFunc(
      module, "analog_runtime_graph_set_dep",
      LLVM::LLVMFunctionType::get(voidType, {ptrType, i32Type, i32Type}, false));
  decls.runtimeInit = getOrCreateExternFunc(
      module, "analog_runtime_init",
      LLVM::LLVMFunctionType::get(ptrType, {ptrType}, false));
  decls.runtimeExecute = getOrCreateExternFunc(
      module, "analog_runtime_execute",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType}, false));
  decls.runtimeDestroy = getOrCreateExternFunc(
      module, "analog_runtime_destroy",
      LLVM::LLVMFunctionType::get(voidType, {ptrType}, false));
  decls.taskArgBuffer = getOrCreateExternFunc(
      module, "analog_runtime_task_arg_buffer",
      LLVM::LLVMFunctionType::get(bufferViewType, {ptrType, i32Type}, false));
  decls.taskSetArgHandle = getOrCreateExternFunc(
      module, "analog_runtime_task_set_arg_handle",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, i32Type, ptrType}, false));
  decls.copyToBuffer = getOrCreateExternFunc(
      module, "analog_runtime_copy_to_buffer",
      LLVM::LLVMFunctionType::get(voidType, {ptrType, ptrType, i64Type}, false));
  decls.persistentHandleCreate = getOrCreateExternFunc(
      module, "analog_runtime_persistent_handle_create",
      LLVM::LLVMFunctionType::get(ptrType, {i64Type, ptrType}, false));
  decls.freeFunc = getOrCreateExternFunc(
      module, "free",
      LLVM::LLVMFunctionType::get(voidType, {ptrType}, false));
  return decls;
}

FailureOr<func::FuncOp> findTaskGraphFunc(ModuleOp module) {
  func::FuncOp taskGraphFunc;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    auto functionType = func.getFunctionType();
    if (functionType.getNumInputs() != 0 || functionType.getNumResults() != 1)
      continue;
    if (!isa<TaskGraphType>(functionType.getResult(0)))
      continue;

    if (taskGraphFunc) {
      module.emitError("expected at most one generated task graph function");
      return failure();
    }
    taskGraphFunc = func;
  }

  return taskGraphFunc;
}

LogicalResult collectGraphModel(ModuleOp module, func::FuncOp taskGraphFunc,
                                GraphModel &model,
                                DenseMap<Value, unsigned> &resourceIndexByValue) {
  model = GraphModel{};
  resourceIndexByValue.clear();

  if (!taskGraphFunc.getBody().hasOneBlock()) {
    taskGraphFunc.emitError("expected generated task graph to have a single block");
    return failure();
  }

  auto workspaceSize = getRequiredI64Attr(taskGraphFunc, kTaskGraphWorkspaceSizeAttrName);
  auto initTaskCount =
      getRequiredI64Attr(taskGraphFunc, kTaskGraphInitTaskCountAttrName);
  if (failed(workspaceSize) || failed(initTaskCount))
    return failure();
  model.workspaceSize = *workspaceSize;
  model.initTaskCount = static_cast<uint32_t>(*initTaskCount);

  for (Operation &op : taskGraphFunc.getBody().front()) {
    std::optional<int32_t> storage;
    Value resourceValue;
    if (auto inputOp = dyn_cast<TaskGraphInputOp>(&op)) {
      storage = STORAGE_INPUT;
      resourceValue = inputOp.getResult();
    } else if (auto outputOp = dyn_cast<TaskGraphOutputOp>(&op)) {
      storage = STORAGE_OUTPUT;
      resourceValue = outputOp.getResult();
    } else if (auto tempOp = dyn_cast<TaskGraphTemporaryOp>(&op)) {
      storage = STORAGE_TEMP;
      resourceValue = tempOp.getResult();
    } else if (auto persistentOp = dyn_cast<TaskGraphPersistentOp>(&op)) {
      storage = STORAGE_PERSISTENT;
      resourceValue = persistentOp.getResult();
    } else {
      continue;
    }

    auto slot = getRequiredI64Attr(&op, kResourceSlotAttrName);
    auto byteSize = getRequiredI64Attr(&op, kResourceByteSizeAttrName);
    if (failed(slot) || failed(byteSize))
      return failure();

    uint64_t workspaceOffsetValue = 0;
    if (*storage == STORAGE_TEMP) {
      auto tempOffset = getRequiredI64Attr(&op, kResourceTempOffsetAttrName);
      if (failed(tempOffset))
        return failure();
      workspaceOffsetValue = *tempOffset;
    }

    auto resourceType = dyn_cast<TaskResourceType>(resourceValue.getType());
    if (!resourceType) {
      op.emitError("expected task graph resource handle type");
      return failure();
    }

    ResourceModel resource;
    resource.value = resourceValue;
    resource.valueType = resourceType.getValueType();
    resource.kind = isHandleResourceType(resource.valueType) ? RES_HANDLE
                                                             : RES_BUFFER;
    resource.storage = *storage;
    resource.slot = static_cast<uint32_t>(*slot);
    resource.byteSize = *byteSize;
    resource.workspaceOffset = workspaceOffsetValue;
    resourceIndexByValue.try_emplace(resourceValue, model.resources.size());
    model.resources.push_back(resource);
  }

  SmallVector<TaskCreateOp> orderedTasks;
  for (Operation &op : taskGraphFunc.getBody().front())
    if (auto taskOp = dyn_cast<TaskCreateOp>(&op))
      orderedTasks.push_back(taskOp);

  std::sort(orderedTasks.begin(), orderedTasks.end(),
            [&](TaskCreateOp lhs, TaskCreateOp rhs) {
              return lookupTaskIndexOrMax(lhs) < lookupTaskIndexOrMax(rhs);
            });

  DenseMap<Value, uint32_t> taskIndexByValue;
  llvm::StringMap<uint32_t> callableIndexBySymbol;

  for (TaskCreateOp taskOp : orderedTasks) {
    auto calleeAttr = taskOp.getCalleeAttr();
    if (!calleeAttr) {
      taskOp.emitError("expected direct task callee symbol");
      return failure();
    }

    auto phase = getTaskPhase(taskOp);
    if (failed(phase))
      return failure();
    auto coreId = getTaskCoreId(taskOp);
    if (failed(coreId))
      return failure();

    uint32_t callableIndex = 0;
    auto existingCallable = callableIndexBySymbol.find(calleeAttr.getValue());
    if (existingCallable == callableIndexBySymbol.end()) {
      auto calleeFunc =
          module.lookupSymbol<LLVM::LLVMFuncOp>(calleeAttr.getValue());
      if (!calleeFunc) {
        taskOp.emitError("expected LLVM callable symbol for task callee '")
            << calleeAttr.getValue()
            << "'; run analog-emit-runtime-graph after convert-func-to-llvm";
        return failure();
      }

      callableIndex = model.callables.size();
      callableIndexBySymbol.try_emplace(calleeAttr.getValue(), callableIndex);
      model.callables.push_back(
          CallableModel{calleeAttr.getValue().str(), calleeFunc, taskOp});
    } else {
      callableIndex = existingCallable->second;
    }

    TaskModel task;
    task.op = taskOp;
    task.callableIndex = callableIndex;
    task.phase = *phase;
    task.coreId = *coreId;
    task.argBegin = model.bindings.size();
    task.depBegin = model.deps.size();

    auto appendBinding = [&](Value resourceValue, uint16_t flags) -> LogicalResult {
      auto resourceIt = resourceIndexByValue.find(resourceValue);
      if (resourceIt == resourceIndexByValue.end()) {
        taskOp.emitError("expected every task resource to carry runtime slot "
                         "metadata");
        return failure();
      }

      const ResourceModel &resource = model.resources[resourceIt->second];
      BindingModel binding;
      binding.kind = resource.kind == RES_HANDLE ? ARG_HANDLE : ARG_BUFFER;
      binding.flags = flags;
      binding.source = SRC_SLOT;
      binding.sourceIndex = resource.slot;
      binding.byteSize = static_cast<uint32_t>(resource.byteSize);
      model.bindings.push_back(binding);
      ++task.argCount;
      return success();
    };

    for (Value input : taskOp.getInputs())
      if (failed(appendBinding(input, ARG_IN)))
        return failure();
    for (Value output : taskOp.getOutputs())
      if (failed(appendBinding(output, ARG_OUT)))
        return failure();

    for (Value dependency : taskOp.getDependencies()) {
      auto dependencyIt = taskIndexByValue.find(dependency);
      if (dependencyIt == taskIndexByValue.end()) {
        taskOp.emitError("expected task dependencies to be emitted in task "
                         "index order");
        return failure();
      }

      model.deps.push_back(dependencyIt->second);
      ++task.depCount;
    }

    taskIndexByValue.try_emplace(taskOp.getResult(), model.tasks.size());
    model.tasks.push_back(task);
  }

  return success();
}

FailureOr<SmallVector<LLVM::LLVMFuncOp>>
emitEntryShims(ModuleOp module, const GraphModel &model,
               const DenseMap<Value, unsigned> &resourceIndexByValue,
               const RuntimeDecls &decls) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  SmallVector<LLVM::LLVMFuncOp> shims;
  Type i32Type = builder.getI32Type();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Location loc = module.getLoc();

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  for (const CallableModel &callable : model.callables) {
    TaskCreateOp representativeTask = callable.representativeTask;
    LLVM::LLVMFuncOp calleeFunc = callable.callee;
    std::string shimName =
        makeUniqueSymbolName(module, "__analog_rt_entry_", callable.symbol);
    auto shimType =
        LLVM::LLVMFunctionType::get(i32Type, {ptrType}, /*isVarArg=*/false);
    auto shim = moduleBuilder.create<LLVM::LLVMFuncOp>(loc, shimName, shimType);
    setPrivateVisibility(shim.getOperation(), builder);
    Block *entryBlock = shim.addEntryBlock(moduleBuilder);

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);
    Value opaque = entryBlock->getArgument(0);

    SmallVector<Value> callOperands;
    unsigned inputArgIndex = 0;
    for (Value input : representativeTask.getInputs()) {
      auto resourceIt = resourceIndexByValue.find(input);
      if (resourceIt == resourceIndexByValue.end()) {
        representativeTask.emitError(
            "expected entry shim inputs to reference known resources");
        return failure();
      }

      const ResourceModel &resource = model.resources[resourceIt->second];
      if (resource.kind == RES_HANDLE) {
        ++inputArgIndex;
        continue;
      }

      auto shapedType = getSupportedBufferType(representativeTask, resource.valueType);
      if (failed(shapedType))
        return failure();

      auto bufferCall = bodyBuilder.create<LLVM::CallOp>(
          loc, decls.taskArgBuffer, ValueRange{opaque,
                                               buildI32Constant(bodyBuilder, loc,
                                                                inputArgIndex)});
      Value bufferView = bufferCall.getResult();
      Value dataPtr = bodyBuilder.create<LLVM::ExtractValueOp>(
          loc, bufferView, ArrayRef<int64_t>{0});
      Value descriptor =
          buildMemRefDescriptor(bodyBuilder, loc, *shapedType, dataPtr);
      flattenMemRefDescriptor(bodyBuilder, loc, descriptor, *shapedType,
                              callOperands);
      ++inputArgIndex;
    }

    auto calleeType = calleeFunc.getFunctionType();
    if (calleeType.getNumParams() != callOperands.size()) {
      representativeTask.emitError(
          "runtime entry shim only supports callables whose LLVM signature "
          "matches the task's buffer inputs");
      return failure();
    }

    auto call = bodyBuilder.create<LLVM::CallOp>(loc, calleeFunc, callOperands);

    SmallVector<std::pair<unsigned, const ResourceModel *>> bufferOutputs;
    SmallVector<unsigned> handleOutputs;
    unsigned outputBindingBase = representativeTask.getInputs().size();
    for (auto indexedOutput : llvm::enumerate(representativeTask.getOutputs())) {
      auto resourceIt = resourceIndexByValue.find(indexedOutput.value());
      if (resourceIt == resourceIndexByValue.end()) {
        representativeTask.emitError(
            "expected entry shim outputs to reference known resources");
        return failure();
      }

      const ResourceModel &resource = model.resources[resourceIt->second];
      unsigned bindingIndex = outputBindingBase + indexedOutput.index();
      if (resource.kind == RES_HANDLE)
        handleOutputs.push_back(bindingIndex);
      else
        bufferOutputs.emplace_back(bindingIndex, &resource);
    }

    if (bufferOutputs.size() > 1) {
      representativeTask.emitError(
          "runtime entry shims currently support at most one buffer output");
      return failure();
    }

    if (bufferOutputs.empty()) {
      if (call.getNumResults() != 0) {
        representativeTask.emitError(
            "expected void LLVM callable for tasks without buffer outputs");
        return failure();
      }
    } else {
      if (call.getNumResults() != 1) {
        representativeTask.emitError(
            "expected single-result LLVM callable for tasks with one buffer "
            "output");
        return failure();
      }

      const ResourceModel &outputResource = *bufferOutputs.front().second;
      Value outputBufferCall = bodyBuilder
                                   .create<LLVM::CallOp>(
                                       loc, decls.taskArgBuffer,
                                       ValueRange{opaque,
                                                  buildI32Constant(
                                                      bodyBuilder, loc,
                                                      bufferOutputs.front()
                                                          .first)})
                                   .getResult();
      Value dstPtr = bodyBuilder.create<LLVM::ExtractValueOp>(
          loc, outputBufferCall, ArrayRef<int64_t>{0});
      Value resultDescriptor = call.getResult();
      Value basePtr = bodyBuilder.create<LLVM::ExtractValueOp>(
          loc, resultDescriptor, ArrayRef<int64_t>{0});
      Value dataPtr = bodyBuilder.create<LLVM::ExtractValueOp>(
          loc, resultDescriptor, ArrayRef<int64_t>{1});
      bodyBuilder.create<LLVM::CallOp>(
          loc, decls.copyToBuffer,
          ValueRange{dstPtr, dataPtr,
                     buildI64Constant(bodyBuilder, loc,
                                      static_cast<int64_t>(outputResource.byteSize))});
      bodyBuilder.create<LLVM::CallOp>(loc, decls.freeFunc, ValueRange{basePtr});
    }

    for (unsigned bindingIndex : handleOutputs) {
      Value handle = bodyBuilder
                         .create<LLVM::CallOp>(
                             loc, decls.persistentHandleCreate,
                             ValueRange{buildI64Constant(bodyBuilder, loc, 0),
                                        buildZeroPointer(bodyBuilder, loc)})
                         .getResult();
      bodyBuilder.create<LLVM::CallOp>(
          loc, decls.taskSetArgHandle,
          ValueRange{opaque,
                     buildI32Constant(bodyBuilder, loc,
                                      static_cast<int32_t>(bindingIndex)),
                     handle});
    }

    bodyBuilder.create<LLVM::ReturnOp>(loc, buildI32Constant(bodyBuilder, loc, 0));
    shims.push_back(shim);
  }

  return shims;
}

LLVM::LLVMFuncOp emitGraphBuilder(ModuleOp module, const GraphModel &model,
                                  ArrayRef<LLVM::LLVMFuncOp> shims,
                                  const RuntimeDecls &decls) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  Location loc = module.getLoc();
  Type ptrType = LLVM::LLVMPointerType::get(context);

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());
  std::string builderName =
      makeUniqueSymbolName(module, "__analog_rt_build_graph_image", "");
  auto builderFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, builderName, LLVM::LLVMFunctionType::get(ptrType, {}, false));
  setPrivateVisibility(builderFunc.getOperation(), builder);

  Block *entryBlock = builderFunc.addEntryBlock(moduleBuilder);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);

  Value graph = bodyBuilder
                    .create<LLVM::CallOp>(
                        loc, decls.graphCreate,
                        ValueRange{
                            buildI32Constant(bodyBuilder, loc,
                                             static_cast<int32_t>(model.resources.size())),
                            buildI32Constant(bodyBuilder, loc,
                                             static_cast<int32_t>(model.callables.size())),
                            buildI32Constant(bodyBuilder, loc,
                                             static_cast<int32_t>(model.tasks.size())),
                            buildI32Constant(bodyBuilder, loc,
                                             static_cast<int32_t>(model.bindings.size())),
                            buildI32Constant(bodyBuilder, loc,
                                             static_cast<int32_t>(model.deps.size())),
                            buildI32Constant(bodyBuilder, loc, 0),
                            buildI32Constant(bodyBuilder, loc, 0),
                            buildI32Constant(bodyBuilder, loc,
                                             static_cast<int32_t>(model.initTaskCount)),
                            buildI64Constant(bodyBuilder, loc,
                                             static_cast<int64_t>(model.workspaceSize))})
                    .getResult();

  for (auto indexedResource : llvm::enumerate(model.resources)) {
    const ResourceModel &resource = indexedResource.value();
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetResource,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedResource.index())),
            buildI32Constant(bodyBuilder, loc, resource.kind),
            buildI32Constant(bodyBuilder, loc, resource.storage),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(resource.slot)),
            buildI64Constant(bodyBuilder, loc,
                             static_cast<int64_t>(resource.byteSize)),
            buildI64Constant(bodyBuilder, loc,
                             static_cast<int64_t>(resource.workspaceOffset))});
  }

  for (auto indexedCallable : llvm::enumerate(shims)) {
    Value fnPtr = bodyBuilder.create<LLVM::AddressOfOp>(loc, indexedCallable.value());
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetCallable,
        ValueRange{graph,
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(indexedCallable.index())),
                   buildI32Constant(bodyBuilder, loc,
                                    static_cast<int32_t>(indexedCallable.index())),
                   fnPtr, buildI32Constant(bodyBuilder, loc, 0)});
  }

  for (auto indexedTask : llvm::enumerate(model.tasks)) {
    const TaskModel &task = indexedTask.value();
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetTask,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedTask.index())),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(task.callableIndex)),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(task.argBegin)),
            buildI16Constant(bodyBuilder, loc, task.argCount),
            buildI16Constant(bodyBuilder, loc, task.phase),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(task.depBegin)),
            buildI16Constant(bodyBuilder, loc, task.depCount),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(task.payloadOffset)),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(task.payloadSize)),
            buildI32Constant(bodyBuilder, loc, task.coreId)});
  }

  for (auto indexedBinding : llvm::enumerate(model.bindings)) {
    const BindingModel &binding = indexedBinding.value();
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetBinding,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedBinding.index())),
            buildI32Constant(bodyBuilder, loc, binding.kind),
            buildI16Constant(bodyBuilder, loc, binding.flags),
            buildI32Constant(bodyBuilder, loc, binding.source),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(binding.sourceIndex)),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(binding.byteOffset)),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(binding.byteSize))});
  }

  for (auto indexedDep : llvm::enumerate(model.deps)) {
    bodyBuilder.create<LLVM::CallOp>(
        loc, decls.graphSetDep,
        ValueRange{
            graph,
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedDep.index())),
            buildI32Constant(bodyBuilder, loc,
                             static_cast<int32_t>(indexedDep.value()))});
  }

  bodyBuilder.create<LLVM::ReturnOp>(loc, graph);
  return builderFunc;
}

LogicalResult emitPublicWrappers(ModuleOp module, LLVM::LLVMFuncOp graphBuilder,
                                 const RuntimeDecls &decls) {
  MLIRContext *context = module.getContext();
  Builder builder(context);
  Location loc = module.getLoc();
  Type ptrType = LLVM::LLVMPointerType::get(context);
  Type i32Type = builder.getI32Type();

  if (module.lookupSymbol("runtime_init") || module.lookupSymbol("runtime_execute") ||
      module.lookupSymbol("runtime_destroy")) {
    module.emitError("expected runtime_init/runtime_execute/runtime_destroy to "
                     "be absent before analog-emit-runtime-graph");
    return failure();
  }

  OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  auto runtimeInitWrapper = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runtime_init",
      LLVM::LLVMFunctionType::get(ptrType, {}, /*isVarArg=*/false));
  auto *initBlock = runtimeInitWrapper.addEntryBlock(moduleBuilder);
  OpBuilder initBuilder = OpBuilder::atBlockBegin(initBlock);
  Value graph = initBuilder.create<LLVM::CallOp>(loc, graphBuilder, ValueRange{}).getResult();
  Value runtime = initBuilder.create<LLVM::CallOp>(loc, decls.runtimeInit,
                                                   ValueRange{graph}).getResult();
  initBuilder.create<LLVM::ReturnOp>(loc, runtime);

  auto runtimeExecuteWrapper = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runtime_execute",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType, ptrType}, false));
  auto *executeBlock = runtimeExecuteWrapper.addEntryBlock(moduleBuilder);
  OpBuilder executeBuilder = OpBuilder::atBlockBegin(executeBlock);
  SmallVector<Value> executeOperands{executeBlock->getArgument(0),
                                     executeBlock->getArgument(1),
                                     executeBlock->getArgument(2)};
  auto executeCall = executeBuilder.create<LLVM::CallOp>(
      loc, decls.runtimeExecute, executeOperands);
  executeBuilder.create<LLVM::ReturnOp>(loc, executeCall.getResult());

  auto runtimeDestroyWrapper = moduleBuilder.create<LLVM::LLVMFuncOp>(
      loc, "runtime_destroy",
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), {ptrType},
                                  false));
  auto *destroyBlock = runtimeDestroyWrapper.addEntryBlock(moduleBuilder);
  OpBuilder destroyBuilder = OpBuilder::atBlockBegin(destroyBlock);
  SmallVector<Value> destroyOperands{destroyBlock->getArgument(0)};
  destroyBuilder.create<LLVM::CallOp>(loc, decls.runtimeDestroy, destroyOperands);
  destroyBuilder.create<LLVM::ReturnOp>(loc, ValueRange{});

  return success();
}

class EmitRuntimeGraphPassImpl final : public EmitRuntimeGraphPass {};

} // namespace

void EmitRuntimeGraphPass::runOnOperation() {
  ModuleOp module = getOperation();

  auto taskGraphFunc = findTaskGraphFunc(module);
  if (failed(taskGraphFunc)) {
    signalPassFailure();
    return;
  }
  if (!*taskGraphFunc)
    return;

  GraphModel model;
  DenseMap<Value, unsigned> resourceIndexByValue;
  if (failed(collectGraphModel(module, *taskGraphFunc, model,
                               resourceIndexByValue))) {
    signalPassFailure();
    return;
  }

  RuntimeDecls decls = getRuntimeDecls(module);
  auto shims =
      emitEntryShims(module, model, resourceIndexByValue, decls);
  if (failed(shims)) {
    signalPassFailure();
    return;
  }

  LLVM::LLVMFuncOp graphBuilder =
      emitGraphBuilder(module, model, *shims, decls);
  if (failed(emitPublicWrappers(module, graphBuilder, decls))) {
    signalPassFailure();
    return;
  }

  // The symbolic task graph has been compiled into runtime metadata and
  // cannot participate in the final LLVM translation.
  taskGraphFunc.value()->erase();
}

void registerEmitRuntimeGraphPass() {
  PassRegistration<EmitRuntimeGraphPass>();
}

} // namespace analog
} // namespace mlir
