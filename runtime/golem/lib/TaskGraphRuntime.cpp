#include "TaskGraphRuntime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <omp.h>

#ifndef NUM_CORES
#error "NUM_CORES must be defined"
#endif

struct RuntimeServices {
  const int32_t *resolved_core_ids = nullptr;
};

struct RuntimeHandle {
  GraphImage *graph = nullptr;
  ExecContext exec{};
  RuntimeTask *runtime_tasks = nullptr;
  TaskCall *task_calls = nullptr;
  int32_t *resolved_core_ids = nullptr;
  RuntimeServices services{};
  uint32_t slot_count = 0;
};

namespace {

constexpr uint32_t kPersistentHandleMagic = 0x414E474C;
constexpr int32_t kRuntimeNumCores = NUM_CORES;
constexpr int32_t kTaskCoreIdNone = -1;
constexpr int32_t kRuntimeErrorInvalidArgument = -1;
constexpr int32_t kRuntimeErrorInvalidGraph = -2;
constexpr int32_t kRuntimeErrorAllocationFailed = -3;
constexpr int32_t kRuntimeErrorInvalidBinding = -4;
constexpr int32_t kRuntimeErrorMissingHandle = -5;
static_assert(kRuntimeNumCores > 0, "NUM_CORES must be positive");

using PersistentHandleDestroyFn = void (*)(void *);

struct PersistentHandleHeader {
  uint32_t magic;
  uint32_t reserved;
  PersistentHandleDestroyFn destroy;
};

template <typename T>
T *allocateZeroedArray(uint32_t count) {
  if (count == 0)
    return nullptr;

  return static_cast<T *>(std::calloc(count, sizeof(T)));
}

void destroyPersistentHandle(void *handle) {
  if (!handle)
    return;

  auto *header = reinterpret_cast<PersistentHandleHeader *>(
      static_cast<uint8_t *>(handle) - sizeof(PersistentHandleHeader));
  if (header->magic != kPersistentHandleMagic) {
    std::free(header);
    return;
  }

  if (header->destroy)
    header->destroy(handle);
  std::free(header);
}

const TaskCall *castTaskCall(void *opaque) {
  return static_cast<const TaskCall *>(opaque);
}

const ArgBinding *taskBinding(const TaskCall *call, uint32_t index) {
  if (!call || !call->graph || !call->task || !call->exec)
    return nullptr;
  if (index >= call->task->arg_count)
    return nullptr;

  uint32_t bindingIndex = call->task->arg_begin + index;
  if (bindingIndex >= call->graph->binding_count)
    return nullptr;
  return &call->graph->bindings[bindingIndex];
}

const uint8_t *bindingData(const TaskCall *call, const ArgBinding *binding) {
  if (!call || !binding || !call->graph)
    return nullptr;

  switch (binding->source) {
  case SRC_INLINE: {
    uint64_t begin =
        static_cast<uint64_t>(call->task->payload_offset) + binding->byte_offset;
    uint64_t end = begin + binding->byte_size;
    if (end > call->graph->payload_blob_size)
      return nullptr;
    return call->graph->payload_blob + begin;
  }
  case SRC_CONST_BLOB: {
    uint64_t begin = static_cast<uint64_t>(binding->index);
    uint64_t end = begin + binding->byte_size;
    if (end > call->graph->const_blob_size)
      return nullptr;
    return call->graph->const_blob + begin;
  }
  default:
    return nullptr;
  }
}

uint32_t computeSlotCount(const GraphImage *graph) {
  uint32_t slotCount = 0;
  if (!graph)
    return slotCount;

  for (uint32_t i = 0; i < graph->resource_count; ++i)
    slotCount = std::max(slotCount, graph->resources[i].slot + 1);
  return slotCount;
}

bool validateGraph(const GraphImage *graph) {
  if (!graph)
    return false;
  if (graph->init_task_count > graph->task_count)
    return false;

  uint32_t slotCount = computeSlotCount(graph);
  for (uint32_t i = 0; i < graph->task_count; ++i) {
    const TaskDesc &task = graph->tasks[i];
    if (task.callable_index >= graph->callable_count)
      return false;
    if (task.arg_begin + task.arg_count > graph->binding_count)
      return false;
    if (task.dep_begin + task.dep_count > graph->dep_count)
      return false;
    if (i < graph->init_task_count) {
      if (task.phase != TASK_PHASE_INIT)
        return false;
    } else if (task.phase != TASK_PHASE_RUN) {
      return false;
    }

    for (uint32_t depIndex = 0; depIndex < task.dep_count; ++depIndex) {
      if (graph->deps[task.dep_begin + depIndex] >= i)
        return false;
    }

    for (uint32_t argIndex = 0; argIndex < task.arg_count; ++argIndex) {
      const ArgBinding &binding = graph->bindings[task.arg_begin + argIndex];
      if (binding.source == SRC_SLOT && binding.index >= slotCount)
        return false;
    }
  }

  return true;
}

void resolveTaskCoreIds(const GraphImage *graph, uint32_t begin, uint32_t end,
                        int32_t *resolved_core_ids) {
  if (!graph || !resolved_core_ids || begin >= end)
    return;

#ifdef ANALOG_RUNTIME_FORWARD_FILL_CORE_IDS
  int32_t next_core = 0;
  bool has_explicit_core = false;
  for (uint32_t i = begin; i < end; ++i) {
    int32_t core_id = graph->tasks[i].core_id;
    if (core_id >= 0) {
      next_core = core_id;
      has_explicit_core = true;
    }
  }

  if (!has_explicit_core) {
    for (uint32_t i = begin; i < end; ++i)
      resolved_core_ids[i] = 0;
    return;
  }

  for (uint32_t i = end; i-- > begin;) {
    int32_t core_id = graph->tasks[i].core_id;
    if (core_id >= 0)
      next_core = core_id;
    resolved_core_ids[i] = (core_id >= 0) ? core_id : next_core;
  }
#else
  int32_t current_core = 0;
  for (uint32_t i = begin; i < end; ++i) {
    int32_t core_id = graph->tasks[i].core_id;
    if (core_id >= 0)
      current_core = core_id;
    resolved_core_ids[i] = (core_id >= 0) ? core_id : current_core;
  }
#endif
}

bool validateResolvedCoreIds(const GraphImage *graph,
                             const int32_t *resolved_core_ids) {
  if (!graph || !resolved_core_ids)
    return false;

  for (uint32_t i = 0; i < graph->task_count; ++i) {
    int32_t core_id = resolved_core_ids[i];
    if (core_id < 0 || core_id >= kRuntimeNumCores)
      return false;
  }

  for (uint32_t i = graph->init_task_count; i < graph->task_count; ++i) {
    const TaskDesc &task = graph->tasks[i];
    for (uint32_t dep_index = 0; dep_index < task.dep_count; ++dep_index) {
      uint32_t dependency = graph->deps[task.dep_begin + dep_index];
      if (dependency >= graph->init_task_count &&
          resolved_core_ids[dependency] != resolved_core_ids[i]) {
        return false;
      }
    }
  }

  return true;
}

void destroyRuntimeHandle(RuntimeHandle *runtime) {
  if (!runtime)
    return;

  if (runtime->graph && runtime->exec.slots) {
    for (uint32_t i = 0; i < runtime->graph->resource_count; ++i) {
      const ResourceDesc &resource = runtime->graph->resources[i];
      if (resource.storage != STORAGE_PERSISTENT || resource.kind != RES_HANDLE ||
          resource.slot >= runtime->slot_count) {
        continue;
      }

      destroyPersistentHandle(runtime->exec.slots[resource.slot].as.handle);
      runtime->exec.slots[resource.slot].as.handle = nullptr;
    }
  }

  std::free(runtime->runtime_tasks);
  std::free(runtime->task_calls);
  std::free(runtime->resolved_core_ids);
  std::free(runtime->exec.slots);
  std::free(runtime->exec.workspace);
  analog_runtime_graph_destroy(runtime->graph);
  std::free(runtime);
}

int32_t runTaskRangeSerial(RuntimeHandle *runtime, uint32_t begin,
                           uint32_t end) {
  if (!runtime || !runtime->runtime_tasks)
    return kRuntimeErrorInvalidArgument;

  for (uint32_t i = begin; i < end; ++i) {
    RuntimeTask &task = runtime->runtime_tasks[i];
    if (!task.fn)
      return kRuntimeErrorInvalidGraph;

    int32_t rc = task.fn(task.opaque);
    if (rc != 0)
      return rc;
  }

  return 0;
}

int32_t runTaskRangeParallel(RuntimeHandle *runtime, uint32_t begin,
                             uint32_t end) {
  if (!runtime || !runtime->runtime_tasks || !runtime->resolved_core_ids)
    return kRuntimeErrorInvalidArgument;

  int32_t thread_errors[kRuntimeNumCores];
  uint32_t failure_task_indices[kRuntimeNumCores];
  for (int32_t i = 0; i < kRuntimeNumCores; ++i) {
    thread_errors[i] = 0;
    failure_task_indices[i] = std::numeric_limits<uint32_t>::max();
  }

#pragma omp parallel num_threads(kRuntimeNumCores)
  {
    int32_t thread_id = omp_get_thread_num();
    int32_t &thread_error = thread_errors[thread_id];
    uint32_t &failure_task_index = failure_task_indices[thread_id];

    for (uint32_t i = begin; i < end; ++i) {
      int32_t tid = runtime->resolved_core_ids[i];
      RuntimeTask &task = runtime->runtime_tasks[i];
      if (thread_id == tid && thread_error == 0) {
        if (!task.fn) {
          thread_error = kRuntimeErrorInvalidGraph;
          failure_task_index = i;
        } else {
          std::fprintf(stderr, "task %u core %d\n", i, tid);
          int32_t rc = task.fn(task.opaque);
          if (rc != 0) {
            thread_error = rc;
            failure_task_index = i;
          }
        }
      }

#pragma omp barrier
    }
  }

  int32_t error_code = 0;
  uint32_t earliest_failure = std::numeric_limits<uint32_t>::max();
  for (int32_t i = 0; i < kRuntimeNumCores; ++i) {
    if (thread_errors[i] != 0 && failure_task_indices[i] < earliest_failure) {
      earliest_failure = failure_task_indices[i];
      error_code = thread_errors[i];
    }
  }

  return error_code;
}

} // namespace

extern "C" {

GraphImage *analog_runtime_graph_create(uint32_t resource_count,
                                        uint32_t callable_count,
                                        uint32_t task_count,
                                        uint32_t binding_count,
                                        uint32_t dep_count,
                                        uint32_t payload_blob_size,
                                        uint32_t const_blob_size,
                                        uint32_t init_task_count,
                                        uint64_t workspace_size) {
  auto *graph = static_cast<GraphImage *>(std::calloc(1, sizeof(GraphImage)));
  if (!graph)
    return nullptr;

  graph->resource_count = resource_count;
  graph->callable_count = callable_count;
  graph->task_count = task_count;
  graph->binding_count = binding_count;
  graph->dep_count = dep_count;
  graph->payload_blob_size = payload_blob_size;
  graph->const_blob_size = const_blob_size;
  graph->init_task_count = init_task_count;
  graph->workspace_size = workspace_size;

  graph->resources = allocateZeroedArray<ResourceDesc>(resource_count);
  graph->callables = allocateZeroedArray<CallableDesc>(callable_count);
  graph->tasks = allocateZeroedArray<TaskDesc>(task_count);
  graph->bindings = allocateZeroedArray<ArgBinding>(binding_count);
  graph->deps = allocateZeroedArray<uint32_t>(dep_count);
  graph->payload_blob = allocateZeroedArray<uint8_t>(payload_blob_size);
  graph->const_blob = allocateZeroedArray<uint8_t>(const_blob_size);

  if ((resource_count && !graph->resources) ||
      (callable_count && !graph->callables) || (task_count && !graph->tasks) ||
      (binding_count && !graph->bindings) || (dep_count && !graph->deps) ||
      (payload_blob_size && !graph->payload_blob) ||
      (const_blob_size && !graph->const_blob)) {
    analog_runtime_graph_destroy(graph);
    return nullptr;
  }

  for (uint32_t i = 0; i < task_count; ++i)
    const_cast<TaskDesc *>(graph->tasks)[i].core_id = kTaskCoreIdNone;

  return graph;
}

void analog_runtime_graph_destroy(GraphImage *graph) {
  if (!graph)
    return;

  std::free(const_cast<ResourceDesc *>(graph->resources));
  std::free(const_cast<CallableDesc *>(graph->callables));
  std::free(const_cast<TaskDesc *>(graph->tasks));
  std::free(const_cast<ArgBinding *>(graph->bindings));
  std::free(const_cast<uint32_t *>(graph->deps));
  std::free(const_cast<uint8_t *>(graph->payload_blob));
  std::free(const_cast<uint8_t *>(graph->const_blob));
  std::free(graph);
}

void analog_runtime_graph_set_resource(GraphImage *graph, uint32_t index,
                                       int32_t kind, int32_t storage,
                                       uint32_t slot, uint64_t byte_size,
                                       uint64_t workspace_offset) {
  if (!graph || index >= graph->resource_count)
    return;

  auto &resource = const_cast<ResourceDesc *>(graph->resources)[index];
  resource.kind = kind;
  resource.storage = storage;
  resource.slot = slot;
  resource.byte_size = byte_size;
  resource.workspace_offset = workspace_offset;
}

void analog_runtime_graph_set_callable(GraphImage *graph, uint32_t index,
                                       uint32_t symbol_id, TaskEntry entry,
                                       uint32_t signature_id) {
  if (!graph || index >= graph->callable_count)
    return;

  auto &callable = const_cast<CallableDesc *>(graph->callables)[index];
  callable.symbol_id = symbol_id;
  callable.entry = entry;
  callable.signature_id = signature_id;
}

void analog_runtime_graph_set_task(GraphImage *graph, uint32_t index,
                                   uint32_t callable_index, uint32_t arg_begin,
                                   uint16_t arg_count, uint16_t phase,
                                   uint32_t dep_begin, uint16_t dep_count,
                                   uint32_t payload_offset,
                                   uint32_t payload_size, int32_t core_id) {
  if (!graph || index >= graph->task_count)
    return;

  auto &task = const_cast<TaskDesc *>(graph->tasks)[index];
  task.callable_index = callable_index;
  task.arg_begin = arg_begin;
  task.arg_count = arg_count;
  task.phase = phase;
  task.dep_begin = dep_begin;
  task.dep_count = dep_count;
  task.payload_offset = payload_offset;
  task.payload_size = payload_size;
  task.core_id = core_id;
}

void analog_runtime_graph_set_binding(GraphImage *graph, uint32_t index,
                                      int32_t kind, uint16_t flags,
                                      int32_t source, uint32_t source_index,
                                      uint32_t byte_offset,
                                      uint32_t byte_size) {
  if (!graph || index >= graph->binding_count)
    return;

  auto &binding = const_cast<ArgBinding *>(graph->bindings)[index];
  binding.kind = kind;
  binding.flags = flags;
  binding.source = source;
  binding.index = source_index;
  binding.byte_offset = byte_offset;
  binding.byte_size = byte_size;
}

void analog_runtime_graph_set_dep(GraphImage *graph, uint32_t index,
                                  uint32_t dependency) {
  if (!graph || index >= graph->dep_count)
    return;

  const_cast<uint32_t *>(graph->deps)[index] = dependency;
}

void analog_runtime_graph_copy_payload(GraphImage *graph, uint32_t offset,
                                       const void *data, uint32_t size) {
  if (!graph || !data || offset + size > graph->payload_blob_size)
    return;

  std::memcpy(const_cast<uint8_t *>(graph->payload_blob) + offset, data, size);
}

void analog_runtime_graph_copy_const_blob(GraphImage *graph, uint32_t offset,
                                          const void *data, uint32_t size) {
  if (!graph || !data || offset + size > graph->const_blob_size)
    return;

  std::memcpy(const_cast<uint8_t *>(graph->const_blob) + offset, data, size);
}

RuntimeHandle *analog_runtime_init(GraphImage *graph) {
  if (!validateGraph(graph))
    return nullptr;

  auto *runtime = static_cast<RuntimeHandle *>(std::calloc(1, sizeof(RuntimeHandle)));
  if (!runtime)
    return nullptr;

  runtime->graph = graph;
  runtime->slot_count = computeSlotCount(graph);
  runtime->exec.workspace = allocateZeroedArray<uint8_t>(
      static_cast<uint32_t>(graph->workspace_size));
  runtime->exec.slots = allocateZeroedArray<SlotValue>(runtime->slot_count);
  runtime->runtime_tasks = allocateZeroedArray<RuntimeTask>(graph->task_count);
  runtime->task_calls = allocateZeroedArray<TaskCall>(graph->task_count);
  runtime->resolved_core_ids = allocateZeroedArray<int32_t>(graph->task_count);
  runtime->services.resolved_core_ids = runtime->resolved_core_ids;
  runtime->exec.services = &runtime->services;

  if ((graph->workspace_size && !runtime->exec.workspace) ||
      (runtime->slot_count && !runtime->exec.slots) ||
      (graph->task_count && !runtime->runtime_tasks) ||
      (graph->task_count && !runtime->task_calls) ||
      (graph->task_count && !runtime->resolved_core_ids)) {
    destroyRuntimeHandle(runtime);
    return nullptr;
  }

  resolveTaskCoreIds(graph, 0, graph->init_task_count, runtime->resolved_core_ids);
  resolveTaskCoreIds(graph, graph->init_task_count, graph->task_count,
                     runtime->resolved_core_ids);
  if (!validateResolvedCoreIds(graph, runtime->resolved_core_ids)) {
    destroyRuntimeHandle(runtime);
    return nullptr;
  }

  for (uint32_t i = 0; i < graph->resource_count; ++i) {
    const ResourceDesc &resource = graph->resources[i];
    if (resource.slot >= runtime->slot_count) {
      destroyRuntimeHandle(runtime);
      return nullptr;
    }

    SlotValue &slot = runtime->exec.slots[resource.slot];
    slot.kind = resource.kind;
    if (resource.kind == RES_BUFFER) {
      slot.as.buffer.data = nullptr;
      slot.as.buffer.byte_size = resource.byte_size;
      if (resource.storage == STORAGE_TEMP) {
        if (resource.workspace_offset + resource.byte_size >
            graph->workspace_size) {
          destroyRuntimeHandle(runtime);
          return nullptr;
        }

        slot.as.buffer.data = runtime->exec.workspace + resource.workspace_offset;
      }
    } else {
      slot.as.handle = nullptr;
    }
  }

  for (uint32_t i = 0; i < graph->task_count; ++i) {
    const TaskDesc &task = graph->tasks[i];
    runtime->runtime_tasks[i].fn = graph->callables[task.callable_index].entry;
    runtime->task_calls[i].graph = graph;
    runtime->task_calls[i].task = &graph->tasks[i];
    runtime->task_calls[i].exec = &runtime->exec;
    runtime->runtime_tasks[i].opaque = &runtime->task_calls[i];
  }

  if (runTaskRangeParallel(runtime, 0, graph->init_task_count) != 0) {
    destroyRuntimeHandle(runtime);
    return nullptr;
  }

  return runtime;
}

int32_t analog_runtime_execute(RuntimeHandle *runtime,
                               const void *const *inputs,
                               void *const *outputs) {
  if (!runtime || !runtime->graph)
    return kRuntimeErrorInvalidArgument;

  uint32_t inputIndex = 0;
  uint32_t outputIndex = 0;
  for (uint32_t i = 0; i < runtime->graph->resource_count; ++i) {
    const ResourceDesc &resource = runtime->graph->resources[i];
    if (resource.slot >= runtime->slot_count)
      return kRuntimeErrorInvalidGraph;

    SlotValue &slot = runtime->exec.slots[resource.slot];
    if (resource.storage == STORAGE_INPUT) {
      if (resource.kind != RES_BUFFER || !inputs || !inputs[inputIndex])
        return kRuntimeErrorInvalidArgument;
      slot.as.buffer.data = const_cast<void *>(inputs[inputIndex++]);
      slot.as.buffer.byte_size = resource.byte_size;
    } else if (resource.storage == STORAGE_OUTPUT) {
      if (resource.kind != RES_BUFFER || !outputs || !outputs[outputIndex])
        return kRuntimeErrorInvalidArgument;
      slot.as.buffer.data = outputs[outputIndex++];
      slot.as.buffer.byte_size = resource.byte_size;
    }
  }

  return runTaskRangeParallel(runtime, runtime->graph->init_task_count,
                              runtime->graph->task_count);
}

void analog_runtime_destroy(RuntimeHandle *runtime) {
  destroyRuntimeHandle(runtime);
}

BufferView analog_runtime_task_arg_buffer(void *opaque, uint32_t index) {
  BufferView empty{nullptr, 0};
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_BUFFER || binding->source != SRC_SLOT)
    return empty;
  if (binding->index >= computeSlotCount(call->graph))
    return empty;

  const SlotValue &slot = call->exec->slots[binding->index];
  if (slot.kind != RES_BUFFER)
    return empty;
  return slot.as.buffer;
}

void *analog_runtime_task_arg_handle(void *opaque, uint32_t index) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_HANDLE || binding->source != SRC_SLOT)
    return nullptr;
  if (binding->index >= computeSlotCount(call->graph))
    return nullptr;

  const SlotValue &slot = call->exec->slots[binding->index];
  if (slot.kind != RES_HANDLE)
    return nullptr;
  return slot.as.handle;
}

int32_t analog_runtime_task_set_arg_handle(void *opaque, uint32_t index,
                                           void *handle) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_HANDLE || binding->source != SRC_SLOT)
    return kRuntimeErrorInvalidBinding;
  if (binding->index >= computeSlotCount(call->graph))
    return kRuntimeErrorInvalidBinding;

  SlotValue &slot = call->exec->slots[binding->index];
  slot.kind = RES_HANDLE;
  slot.as.handle = handle;
  return 0;
}

int32_t analog_runtime_task_require_handle(void *opaque, uint32_t index) {
  return analog_runtime_task_arg_handle(opaque, index) ? 0
                                                       : kRuntimeErrorMissingHandle;
}

int32_t analog_runtime_task_arg_i32(void *opaque, uint32_t index) {
  int32_t value = 0;
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  const uint8_t *data = bindingData(call, binding);
  if (!binding || binding->kind != ARG_I32 || !data || binding->byte_size < sizeof(value))
    return value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

int64_t analog_runtime_task_arg_i64(void *opaque, uint32_t index) {
  int64_t value = 0;
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  const uint8_t *data = bindingData(call, binding);
  if (!binding || binding->kind != ARG_I64 || !data || binding->byte_size < sizeof(value))
    return value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

float analog_runtime_task_arg_f32(void *opaque, uint32_t index) {
  float value = 0.0f;
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  const uint8_t *data = bindingData(call, binding);
  if (!binding || binding->kind != ARG_F32 || !data || binding->byte_size < sizeof(value))
    return value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

double analog_runtime_task_arg_f64(void *opaque, uint32_t index) {
  double value = 0.0;
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  const uint8_t *data = bindingData(call, binding);
  if (!binding || binding->kind != ARG_F64 || !data || binding->byte_size < sizeof(value))
    return value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

const void *analog_runtime_task_arg_bytes(void *opaque, uint32_t index) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding || binding->kind != ARG_BYTES)
    return nullptr;
  return bindingData(call, binding);
}

uint32_t analog_runtime_task_arg_size(void *opaque, uint32_t index) {
  const TaskCall *call = castTaskCall(opaque);
  const ArgBinding *binding = taskBinding(call, index);
  if (!binding)
    return 0;
  return binding->byte_size;
}

int32_t analog_runtime_task_core_id(void *opaque) {
  const TaskCall *call = castTaskCall(opaque);
  if (!call || !call->task)
    return kTaskCoreIdNone;

  const RuntimeServices *services = nullptr;
  if (call->exec)
    services = static_cast<const RuntimeServices *>(call->exec->services);
  if (!services || !services->resolved_core_ids || !call->graph ||
      call->task < call->graph->tasks) {
    return call->task->core_id;
  }

  uint32_t task_index = static_cast<uint32_t>(call->task - call->graph->tasks);
  if (task_index >= call->graph->task_count)
    return call->task->core_id;
  return services->resolved_core_ids[task_index];
}

void analog_runtime_copy_to_buffer(void *dst, const void *src, uint64_t size) {
  if (!dst || !src || size == 0)
    return;
  std::memcpy(dst, src, static_cast<size_t>(size));
}

void *analog_runtime_persistent_handle_create(uint64_t payload_size,
                                              void *destroy_fn) {
  size_t allocationSize =
      sizeof(PersistentHandleHeader) + static_cast<size_t>(payload_size);
  auto *header =
      static_cast<PersistentHandleHeader *>(std::calloc(1, allocationSize));
  if (!header)
    return nullptr;

  header->magic = kPersistentHandleMagic;
  header->destroy = reinterpret_cast<PersistentHandleDestroyFn>(destroy_fn);
  return reinterpret_cast<uint8_t *>(header) + sizeof(PersistentHandleHeader);
}

} // extern "C"
