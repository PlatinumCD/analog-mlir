#include "python_bridge.h"

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <optional>
#include <string>
#include <unistd.h>

namespace {

struct _object;
using PyObject = _object;
using PyGILState_STATE = int;

struct PythonApi {
  void *handle = nullptr;
  int (*Py_IsInitialized)() = nullptr;
  void (*Py_Initialize)() = nullptr;
  PyGILState_STATE (*PyGILState_Ensure)() = nullptr;
  void (*PyGILState_Release)(PyGILState_STATE) = nullptr;
  void *(*PyEval_SaveThread)() = nullptr;
  void (*PyErr_Print)() = nullptr;
  PyObject *(*PySys_GetObject)(const char *) = nullptr;
  PyObject *(*PyUnicode_FromString)(const char *) = nullptr;
  int (*PyList_Append)(PyObject *, PyObject *) = nullptr;
  PyObject *(*PyImport_ImportModule)(const char *) = nullptr;
  PyObject *(*PyObject_GetAttrString)(PyObject *, const char *) = nullptr;
  PyObject *(*PyObject_CallObject)(PyObject *, PyObject *) = nullptr;
  PyObject *(*PyTuple_New)(ssize_t) = nullptr;
  int (*PyTuple_SetItem)(PyObject *, ssize_t, PyObject *) = nullptr;
  PyObject *(*PyLong_FromLong)(long) = nullptr;
  PyObject *(*PyLong_FromVoidPtr)(void *) = nullptr;
  int (*PyBytes_AsStringAndSize)(PyObject *, char **, ssize_t *) = nullptr;
  void (*Py_DecRef)(PyObject *) = nullptr;
};

struct PythonBridgeState {
  PythonApi api;
  bool moduleInitialized = false;
  bool releasedMainThread = false;
  PyObject *coreManagerModule = nullptr;
  PyObject *coreManagerRuntime = nullptr;
  PyObject *initializeFn = nullptr;
  PyObject *getCoreManagerFn = nullptr;
};

PythonBridgeState &getState() {
  static PythonBridgeState state;
  return state;
}

template <typename T>
bool loadSymbol(PythonApi &api, T &fn, const char *name) {
  fn = reinterpret_cast<T>(dlsym(api.handle, name));
  if (!fn) {
    std::fprintf(stderr, "[python bridge] missing CPython symbol: %s\n", name);
    return false;
  }
  return true;
}

bool loadPythonApi() {
  PythonBridgeState &state = getState();
  if (state.api.handle) {
    return true;
  }

  static const char *kPythonLibCandidates[] = {
      "libpython3.10.so.1.0",
      "libpython3.10.so.1",
      "libpython3.10.so",
      "libpython3.11.so.1.0",
      "libpython3.11.so.1",
      "libpython3.11.so",
  };

  for (const char *candidate : kPythonLibCandidates) {
    state.api.handle = dlopen(candidate, RTLD_NOW | RTLD_GLOBAL);
    if (state.api.handle) {
      break;
    }
  }

  if (!state.api.handle) {
    std::fprintf(stderr, "[python bridge] failed to load libpython runtime\n");
    return false;
  }

  return loadSymbol(state.api, state.api.Py_IsInitialized, "Py_IsInitialized") &&
         loadSymbol(state.api, state.api.Py_Initialize, "Py_Initialize") &&
         loadSymbol(state.api, state.api.PyGILState_Ensure, "PyGILState_Ensure") &&
         loadSymbol(state.api, state.api.PyGILState_Release, "PyGILState_Release") &&
         loadSymbol(state.api, state.api.PyEval_SaveThread, "PyEval_SaveThread") &&
         loadSymbol(state.api, state.api.PyErr_Print, "PyErr_Print") &&
         loadSymbol(state.api, state.api.PySys_GetObject, "PySys_GetObject") &&
         loadSymbol(state.api, state.api.PyUnicode_FromString, "PyUnicode_FromString") &&
         loadSymbol(state.api, state.api.PyList_Append, "PyList_Append") &&
         loadSymbol(state.api, state.api.PyImport_ImportModule, "PyImport_ImportModule") &&
         loadSymbol(state.api, state.api.PyObject_GetAttrString,
                    "PyObject_GetAttrString") &&
         loadSymbol(state.api, state.api.PyObject_CallObject,
                    "PyObject_CallObject") &&
         loadSymbol(state.api, state.api.PyTuple_New, "PyTuple_New") &&
         loadSymbol(state.api, state.api.PyTuple_SetItem, "PyTuple_SetItem") &&
         loadSymbol(state.api, state.api.PyLong_FromLong, "PyLong_FromLong") &&
         loadSymbol(state.api, state.api.PyLong_FromVoidPtr, "PyLong_FromVoidPtr") &&
         loadSymbol(state.api, state.api.PyBytes_AsStringAndSize,
                    "PyBytes_AsStringAndSize") &&
         loadSymbol(state.api, state.api.Py_DecRef, "Py_DecRef");
}

std::optional<std::filesystem::path> getExecutablePath() {
  char buffer[4096];
  ssize_t size = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (size < 0) {
    return std::nullopt;
  }
  buffer[size] = '\0';
  return std::filesystem::path(buffer);
}

std::optional<std::filesystem::path> getPythonModuleDir() {
  auto exePath = getExecutablePath();
  if (!exePath) {
    return std::nullopt;
  }

  std::filesystem::path dir = exePath->parent_path();
  for (int i = 0; i < 3; ++i) {
    dir = dir.parent_path();
  }
  return dir / "python";
}

bool appendPythonPath(const std::filesystem::path &path) {
  PythonBridgeState &state = getState();
  PyObject *sysPath = state.api.PySys_GetObject("path");
  if (!sysPath) {
    state.api.PyErr_Print();
    return false;
  }

  PyObject *pathObj = state.api.PyUnicode_FromString(path.string().c_str());
  if (!pathObj) {
    state.api.PyErr_Print();
    return false;
  }

  int rc = state.api.PyList_Append(sysPath, pathObj);
  state.api.Py_DecRef(pathObj);
  if (rc != 0) {
    state.api.PyErr_Print();
    return false;
  }
  return true;
}

long readEnvInt(const char *name, long fallback) {
  if (const char *value = std::getenv(name)) {
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end != value) {
      return parsed;
    }
  }
  return fallback;
}

} // namespace

bool analog_debug_python_bridge_initialize() {
  PythonBridgeState &state = getState();
  if (state.moduleInitialized) {
    return true;
  }

  if (!loadPythonApi()) {
    return false;
  }

  if (!state.api.Py_IsInitialized()) {
    state.api.Py_Initialize();
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  auto pythonDir = getPythonModuleDir();
  if (!pythonDir || !std::filesystem::exists(*pythonDir)) {
    std::fprintf(stderr, "[python bridge] python module directory not found\n");
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (!appendPythonPath(*pythonDir)) {
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.coreManagerModule = state.api.PyImport_ImportModule("core_manager");
  if (!state.coreManagerModule) {
    std::fprintf(stderr, "[python bridge] failed to import core_manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.coreManagerRuntime =
      state.api.PyObject_GetAttrString(state.coreManagerModule,
                                       "CoreManagerRuntime");
  if (!state.coreManagerRuntime) {
    std::fprintf(stderr, "[python bridge] core_manager missing CoreManagerRuntime\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.initializeFn =
      state.api.PyObject_GetAttrString(state.coreManagerRuntime, "initialize");
  if (!state.initializeFn) {
    std::fprintf(stderr, "[python bridge] CoreManagerRuntime missing initialize\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.getCoreManagerFn =
      state.api.PyObject_GetAttrString(state.coreManagerRuntime, "get_core_manager");
  if (!state.getCoreManagerFn) {
    std::fprintf(stderr,
                 "[python bridge] CoreManagerRuntime missing get_core_manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(4);
  if (!args) {
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  const long numCores = readEnvInt("NUM_CORES", 1);
  const long arraysPerCore = readEnvInt("ARRAYS_PER_CORE", 1);
  const long arrayRows = readEnvInt("ARRAY_ROWS", 1);
  const long arrayCols = readEnvInt("ARRAY_COLS", 1);

  if (state.api.PyTuple_SetItem(args, 0, state.api.PyLong_FromLong(numCores)) != 0 ||
      state.api.PyTuple_SetItem(args, 1,
                                state.api.PyLong_FromLong(arraysPerCore)) != 0 ||
      state.api.PyTuple_SetItem(args, 2, state.api.PyLong_FromLong(arrayRows)) != 0 ||
      state.api.PyTuple_SetItem(args, 3, state.api.PyLong_FromLong(arrayCols)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *coreManager = state.api.PyObject_CallObject(state.initializeFn, args);
  state.api.Py_DecRef(args);
  if (!coreManager) {
    std::fprintf(stderr,
                 "[python bridge] failed to initialize CoreManagerRuntime\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }
  state.api.Py_DecRef(coreManager);

  state.moduleInitialized = true;

  if (!state.releasedMainThread) {
    state.api.PyEval_SaveThread();
    state.releasedMainThread = true;
    return true;
  }

  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_dispatch_weight(int32_t weightId) {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *setActiveCoreFn =
      state.api.PyObject_GetAttrString(coreManager, "set_active_core");
  if (!setActiveCoreFn) {
    std::fprintf(stderr, "[python bridge] core manager missing set_active_core\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(2);
  if (!args) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(setActiveCoreFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0,
                                state.api.PyLong_FromLong(weightId)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.Py_DecRef(setActiveCoreFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(setActiveCoreFn, args);
  state.api.Py_DecRef(args);
  state.api.Py_DecRef(setActiveCoreFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to set active core\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_wait_weights() {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *clearActiveCoreFn =
      state.api.PyObject_GetAttrString(coreManager, "clear_active_core");
  if (!clearActiveCoreFn) {
    std::fprintf(stderr, "[python bridge] core manager missing clear_active_core\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(clearActiveCoreFn, nullptr);
  state.api.Py_DecRef(clearActiveCoreFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to clear active core\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_dispatch_layer(int32_t layerId) {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *setActiveCoreFn =
      state.api.PyObject_GetAttrString(coreManager, "set_active_core");
  if (!setActiveCoreFn) {
    std::fprintf(stderr, "[python bridge] core manager missing set_active_core\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(1);
  if (!args) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(setActiveCoreFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0,
                                state.api.PyLong_FromLong(layerId)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.Py_DecRef(setActiveCoreFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(setActiveCoreFn, args);
  state.api.Py_DecRef(args);
  state.api.Py_DecRef(setActiveCoreFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to set active core\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_wait_layers() { return true; }

bool analog_debug_python_bridge_record_mvm_set(void *data, int32_t rawArrayId) {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *recordMvmSetFn =
      state.api.PyObject_GetAttrString(coreManager, "record_mvm_set");
  if (!recordMvmSetFn) {
    std::fprintf(stderr, "[python bridge] core manager missing record_mvm_set\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(1);
  if (!args) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(recordMvmSetFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0, state.api.PyLong_FromVoidPtr(data)) != 0 ||
      state.api.PyTuple_SetItem(args, 1,
                                state.api.PyLong_FromLong(rawArrayId)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.Py_DecRef(recordMvmSetFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(recordMvmSetFn, args);
  state.api.Py_DecRef(args);
  state.api.Py_DecRef(recordMvmSetFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to record mvm_set\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_record_mvm_load(void *data, int32_t rawArrayId) {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *recordMvmLoadFn =
      state.api.PyObject_GetAttrString(coreManager, "record_mvm_load");
  if (!recordMvmLoadFn) {
    std::fprintf(stderr, "[python bridge] core manager missing record_mvm_load\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(2);
  if (!args) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(recordMvmLoadFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0, state.api.PyLong_FromVoidPtr(data)) != 0 ||
      state.api.PyTuple_SetItem(args, 1,
                                state.api.PyLong_FromLong(rawArrayId)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.Py_DecRef(recordMvmLoadFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(recordMvmLoadFn, args);
  state.api.Py_DecRef(args);
  state.api.Py_DecRef(recordMvmLoadFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to record mvm_load\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_record_mvm_compute(int32_t rawArrayId) {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *recordMvmComputeFn =
      state.api.PyObject_GetAttrString(coreManager, "record_mvm_compute");
  if (!recordMvmComputeFn) {
    std::fprintf(stderr,
                 "[python bridge] core manager missing record_mvm_compute\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(1);
  if (!args) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(recordMvmComputeFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0,
                                state.api.PyLong_FromLong(rawArrayId)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.Py_DecRef(recordMvmComputeFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(recordMvmComputeFn, args);
  state.api.Py_DecRef(args);
  state.api.Py_DecRef(recordMvmComputeFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to record mvm_compute\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}

bool analog_debug_python_bridge_record_mvm_store(void *data, int32_t rawArrayId) {
  PythonBridgeState &state = getState();
  if (!analog_debug_python_bridge_initialize()) {
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *coreManager =
      state.api.PyObject_CallObject(state.getCoreManagerFn, nullptr);
  if (!coreManager) {
    std::fprintf(stderr, "[python bridge] failed to fetch core manager\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *recordMvmStoreFn =
      state.api.PyObject_GetAttrString(coreManager, "record_mvm_store");
  if (!recordMvmStoreFn) {
    std::fprintf(stderr, "[python bridge] core manager missing record_mvm_store\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *args = state.api.PyTuple_New(2);
  if (!args) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(recordMvmStoreFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0,
                                state.api.PyLong_FromLong(rawArrayId)) != 0) {
    state.api.PyErr_Print();
    state.api.Py_DecRef(args);
    state.api.Py_DecRef(recordMvmStoreFn);
    state.api.Py_DecRef(coreManager);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(recordMvmStoreFn, args);
  state.api.Py_DecRef(args);
  state.api.Py_DecRef(recordMvmStoreFn);
  state.api.Py_DecRef(coreManager);
  if (!result) {
    std::fprintf(stderr, "[python bridge] failed to record mvm_store\n");
    state.api.PyErr_Print();
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *toBytesFn = state.api.PyObject_GetAttrString(result, "tobytes");
  if (!toBytesFn) {
    std::fprintf(stderr, "[python bridge] mvm_store result missing tobytes\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(result);
    state.api.PyGILState_Release(gil);
    return false;
  }

  PyObject *bytesObject = state.api.PyObject_CallObject(toBytesFn, nullptr);
  state.api.Py_DecRef(toBytesFn);
  if (!bytesObject) {
    std::fprintf(stderr, "[python bridge] failed to serialize mvm_store result\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(result);
    state.api.PyGILState_Release(gil);
    return false;
  }

  char *bytesData = nullptr;
  ssize_t bytesSize = 0;
  if (state.api.PyBytes_AsStringAndSize(bytesObject, &bytesData, &bytesSize) != 0) {
    std::fprintf(stderr, "[python bridge] failed to read mvm_store bytes\n");
    state.api.PyErr_Print();
    state.api.Py_DecRef(bytesObject);
    state.api.Py_DecRef(result);
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (bytesSize > 0) {
    std::memcpy(data, bytesData, static_cast<size_t>(bytesSize));
  }

  state.api.Py_DecRef(bytesObject);
  state.api.Py_DecRef(result);
  state.api.PyGILState_Release(gil);
  return true;
}
