#include "../../headers/python_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <optional>
#include <string>
#include <unistd.h>

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

#if NUM_LAYERS <= 0
#error "NUM_LAYERS must be > 0"
#endif

namespace {

struct _object;
using PyObject = _object;
using PyGILState_STATE = int;
using PyThreadState = void;

struct PythonApi {
  void *handle = nullptr;

  int (*Py_IsInitialized)() = nullptr;
  void (*Py_Initialize)() = nullptr;
  PyGILState_STATE (*PyGILState_Ensure)() = nullptr;
  void (*PyGILState_Release)(PyGILState_STATE) = nullptr;
  PyThreadState *(*PyEval_SaveThread)() = nullptr;
  PyObject *(*PyErr_Occurred)() = nullptr;
  void (*PyErr_Print)() = nullptr;
  PyObject *(*PySys_GetObject)(const char *) = nullptr;
  PyObject *(*PyUnicode_FromString)(const char *) = nullptr;
  int (*PyList_Append)(PyObject *, PyObject *) = nullptr;
  PyObject *(*PyImport_ImportModule)(const char *) = nullptr;
  PyObject *(*PyObject_GetAttrString)(PyObject *, const char *) = nullptr;
  PyObject *(*PyObject_CallObject)(PyObject *, PyObject *) = nullptr;
  int (*PyObject_IsTrue)(PyObject *) = nullptr;
  PyObject *(*PyTuple_New)(ssize_t) = nullptr;
  int (*PyTuple_SetItem)(PyObject *, ssize_t, PyObject *) = nullptr;
  PyObject *(*PyLong_FromLong)(long) = nullptr;
  void (*Py_DecRef)(PyObject *) = nullptr;
};

struct PythonBridgeState {
  PythonApi api;
  bool interpreterInitialized = false;
  bool moduleInitialized = false;
  bool releasedMainThread = false;
  PyObject *module = nullptr;
  PyObject *initializeFn = nullptr;
  PyObject *bindFn = nullptr;
  PyObject *dispatchFn = nullptr;
  PyObject *waitFn = nullptr;
  std::string boundSoPath;
};

PythonBridgeState &getState() {
  static PythonBridgeState state;
  return state;
}

void clearPythonError(const char *context) {
  PythonBridgeState &state = getState();
  if (!state.api.PyErr_Occurred || !state.api.PyErr_Occurred()) {
    return;
  }
  std::fprintf(stderr, "[python bridge] %s failed\n", context);
  if (state.api.PyErr_Print) {
    state.api.PyErr_Print();
  }
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
      "/usr/lib/python3.10/config-3.10-aarch64-linux-gnu/libpython3.10.so",
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
         loadSymbol(state.api, state.api.PyErr_Occurred, "PyErr_Occurred") &&
         loadSymbol(state.api, state.api.PyErr_Print, "PyErr_Print") &&
         loadSymbol(state.api, state.api.PySys_GetObject, "PySys_GetObject") &&
         loadSymbol(state.api, state.api.PyUnicode_FromString, "PyUnicode_FromString") &&
         loadSymbol(state.api, state.api.PyList_Append, "PyList_Append") &&
         loadSymbol(state.api, state.api.PyImport_ImportModule, "PyImport_ImportModule") &&
         loadSymbol(state.api, state.api.PyObject_GetAttrString, "PyObject_GetAttrString") &&
         loadSymbol(state.api, state.api.PyObject_CallObject, "PyObject_CallObject") &&
         loadSymbol(state.api, state.api.PyObject_IsTrue, "PyObject_IsTrue") &&
         loadSymbol(state.api, state.api.PyTuple_New, "PyTuple_New") &&
         loadSymbol(state.api, state.api.PyTuple_SetItem, "PyTuple_SetItem") &&
         loadSymbol(state.api, state.api.PyLong_FromLong, "PyLong_FromLong") &&
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

std::optional<std::filesystem::path> getCurrentSharedObjectPath() {
  auto exePath = getExecutablePath();
  if (!exePath) {
    return std::nullopt;
  }

  std::filesystem::path outDir = exePath->parent_path();
  std::string testName = exePath->stem().string();
  return outDir / ("libpython_" + testName + ".so");
}

bool appendPythonPath(PyObject *sysPath, const std::filesystem::path &path) {
  PythonBridgeState &state = getState();
  PyObject *pathObj = state.api.PyUnicode_FromString(path.string().c_str());
  if (!pathObj) {
    clearPythonError("creating Python path string");
    return false;
  }

  int rc = state.api.PyList_Append(sysPath, pathObj);
  state.api.Py_DecRef(pathObj);
  if (rc != 0) {
    clearPythonError("appending module path to sys.path");
    return false;
  }
  return true;
}

bool callBridgeBoolNoArgs(PyObject *callable, const char *context) {
  PythonBridgeState &state = getState();
  PyObject *result = state.api.PyObject_CallObject(callable, nullptr);
  if (!result) {
    clearPythonError(context);
    return false;
  }

  int ok = state.api.PyObject_IsTrue(result);
  state.api.Py_DecRef(result);
  return ok == 1;
}

bool callBridgeBoolOneLong(PyObject *callable, long value, const char *context) {
  PythonBridgeState &state = getState();
  PyObject *args = state.api.PyTuple_New(1);
  if (!args) {
    clearPythonError("allocating Python tuple");
    return false;
  }

  if (state.api.PyTuple_SetItem(args, 0, state.api.PyLong_FromLong(value)) != 0) {
    clearPythonError(context);
    state.api.Py_DecRef(args);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(callable, args);
  state.api.Py_DecRef(args);
  if (!result) {
    clearPythonError(context);
    return false;
  }

  int ok = state.api.PyObject_IsTrue(result);
  state.api.Py_DecRef(result);
  return ok == 1;
}

bool callBridgeBoolOneString(PyObject *callable, const std::string &value,
                             const char *context) {
  PythonBridgeState &state = getState();
  PyObject *args = state.api.PyTuple_New(1);
  if (!args) {
    clearPythonError("allocating Python tuple");
    return false;
  }

  if (state.api.PyTuple_SetItem(
          args, 0, state.api.PyUnicode_FromString(value.c_str())) != 0) {
    clearPythonError(context);
    state.api.Py_DecRef(args);
    return false;
  }

  PyObject *result = state.api.PyObject_CallObject(callable, args);
  state.api.Py_DecRef(args);
  if (!result) {
    clearPythonError(context);
    return false;
  }

  int ok = state.api.PyObject_IsTrue(result);
  state.api.Py_DecRef(result);
  return ok == 1;
}

bool initializeInterpreterAndModule() {
  PythonBridgeState &state = getState();
  if (state.moduleInitialized) {
    return true;
  }

  if (!loadPythonApi()) {
    return false;
  }

  if (!state.api.Py_IsInitialized()) {
    state.api.Py_Initialize();
    state.interpreterInitialized = true;
  }

  auto pythonDir = getPythonModuleDir();
  if (!pythonDir || !std::filesystem::exists(*pythonDir)) {
    std::fprintf(stderr, "[python bridge] python module directory not found\n");
    return false;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();

  PyObject *sysPath = state.api.PySys_GetObject("path");
  if (!sysPath || !appendPythonPath(sysPath, *pythonDir)) {
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.module = state.api.PyImport_ImportModule("debug_weight_bridge");
  if (!state.module) {
    clearPythonError("importing debug_weight_bridge");
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.initializeFn =
      state.api.PyObject_GetAttrString(state.module, "initialize_bridge");
  state.bindFn = state.api.PyObject_GetAttrString(state.module, "bind_shared_object");
  state.dispatchFn = state.api.PyObject_GetAttrString(state.module, "dispatch_weight");
  state.waitFn = state.api.PyObject_GetAttrString(state.module, "wait_weights");

  if (!state.initializeFn || !state.bindFn || !state.dispatchFn || !state.waitFn) {
    clearPythonError("loading debug_weight_bridge callables");
    state.api.PyGILState_Release(gil);
    return false;
  }

  if (!callBridgeBoolOneLong(state.initializeFn, NUM_LAYERS,
                             "calling initialize_bridge")) {
    state.api.PyGILState_Release(gil);
    return false;
  }

  state.moduleInitialized = true;

  if (!state.releasedMainThread) {
    state.api.PyEval_SaveThread();
    state.releasedMainThread = true;
    return true;
  }

  state.api.PyGILState_Release(gil);
  return true;
}

} // namespace

bool analog_debug_python_bridge_initialize() {
  return initializeInterpreterAndModule();
}

bool analog_debug_python_bridge_bind_current_test() {
  if (!initializeInterpreterAndModule()) {
    return false;
  }

  auto soPath = getCurrentSharedObjectPath();
  if (!soPath) {
    std::fprintf(stderr,
                 "[python bridge] failed to derive current test shared object path\n");
    return false;
  }

  if (!std::filesystem::exists(*soPath)) {
    std::fprintf(stderr, "[python bridge] shared object not found: %s\n",
                 soPath->c_str());
    return false;
  }

  PythonBridgeState &state = getState();
  if (state.boundSoPath == soPath->string()) {
    return true;
  }

  PyGILState_STATE gil = state.api.PyGILState_Ensure();
  bool ok =
      callBridgeBoolOneString(state.bindFn, soPath->string(),
                              "calling bind_shared_object");
  state.api.PyGILState_Release(gil);
  if (!ok) {
    return false;
  }

  state.boundSoPath = soPath->string();
  return true;
}

bool analog_debug_python_bridge_dispatch_weight(int32_t weightId) {
  if (!analog_debug_python_bridge_bind_current_test()) {
    return false;
  }

  PythonBridgeState &state = getState();
  PyGILState_STATE gil = state.api.PyGILState_Ensure();
  bool ok = callBridgeBoolOneLong(state.dispatchFn, weightId,
                                  "calling dispatch_weight");
  state.api.PyGILState_Release(gil);
  return ok;
}

bool analog_debug_python_bridge_wait_weights() {
  if (!analog_debug_python_bridge_bind_current_test()) {
    return false;
  }

  PythonBridgeState &state = getState();
  PyGILState_STATE gil = state.api.PyGILState_Ensure();
  bool ok = callBridgeBoolNoArgs(state.waitFn, "calling wait_weights");
  state.api.PyGILState_Release(gil);
  return ok;
}
