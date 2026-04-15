#include "SimBridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <sys/types.h>
#include <mutex>
#include <string>

using Py_ssize_t = ssize_t;
using PyGILState_STATE = int;

typedef struct _object PyObject;

struct Py_buffer {
  void *buf;
  PyObject *obj;
  Py_ssize_t len;
  Py_ssize_t itemsize;
  int readonly;
  int ndim;
  char *format;
  Py_ssize_t *shape;
  Py_ssize_t *strides;
  Py_ssize_t *suboffsets;
  void *internal;
};

extern "C" {
int Py_IsInitialized(void);
void Py_Initialize(void);
PyGILState_STATE PyGILState_Ensure(void);
void PyGILState_Release(PyGILState_STATE);
PyObject *PyErr_Occurred(void);
void PyErr_Fetch(PyObject **, PyObject **, PyObject **);
void PyErr_NormalizeException(PyObject **, PyObject **, PyObject **);
PyObject *PyObject_Str(PyObject *);
const char *PyUnicode_AsUTF8(PyObject *);
PyObject *PySys_GetObject(const char *);
int PySequence_Contains(PyObject *, PyObject *);
int PyList_Append(PyObject *, PyObject *);
PyObject *PyImport_ImportModule(const char *);
PyObject *PyObject_GetAttrString(PyObject *, const char *);
PyObject *PyTuple_New(Py_ssize_t);
int PyTuple_SetItem(PyObject *, Py_ssize_t, PyObject *);
PyObject *PyLong_FromLong(long);
PyObject *PyLong_FromVoidPtr(void *);
PyObject *PyObject_CallObject(PyObject *, PyObject *);
int PyObject_GetBuffer(PyObject *, Py_buffer *, int);
void PyBuffer_Release(Py_buffer *);
PyObject *PyUnicode_FromString(const char *);
void Py_IncRef(PyObject *);
void Py_DecRef(PyObject *);
}

namespace analog::golem_sim {

namespace {

constexpr int kPyBufferSimple = 0;

struct BridgeState {
  bool initialized = false;
  std::int32_t clients = 0;
  PyObject *runtimeClass = nullptr;
  PyObject *manager = nullptr;
};

BridgeState &getState() {
  static BridgeState state;
  return state;
}

std::mutex &getMutex() {
  static std::mutex mutex;
  return mutex;
}

void decref(PyObject *object) {
  if (object)
    Py_DecRef(object);
}

[[noreturn]] void fail(const char *context, const std::string &detail) {
  std::fprintf(stderr, "golem-simulator bridge error: %s", context);
  if (!detail.empty())
    std::fprintf(stderr, ": %s", detail.c_str());
  std::fprintf(stderr, "\n");
  std::abort();
}

std::string fetchPythonError() {
  if (!PyErr_Occurred())
    return {};

  PyObject *type = nullptr;
  PyObject *value = nullptr;
  PyObject *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);

  std::string message = "unknown Python error";
  if (value) {
    PyObject *valueString = PyObject_Str(value);
    if (valueString) {
      const char *utf8 = PyUnicode_AsUTF8(valueString);
      if (utf8)
        message = utf8;
      decref(valueString);
    }
  }

  decref(type);
  decref(value);
  decref(traceback);
  return message;
}

PyObject *importModuleWithPath(const char *path, const char *moduleName) {
  PyObject *sysPath = PySys_GetObject("path");
  if (!sysPath)
    fail("load sys.path", "sys.path is unavailable");

  PyObject *pathString = PyUnicode_FromString(path);
  if (!pathString)
    fail("build Python path string", fetchPythonError());

  int contains = PySequence_Contains(sysPath, pathString);
  if (contains < 0) {
    decref(pathString);
    fail("inspect sys.path", fetchPythonError());
  }
  if (contains == 0 && PyList_Append(sysPath, pathString) != 0) {
    decref(pathString);
    fail("append simulator path to sys.path", fetchPythonError());
  }
  decref(pathString);

  PyObject *module = PyImport_ImportModule(moduleName);
  if (!module)
    fail("import Python simulator module", fetchPythonError());
  return module;
}

std::int32_t resolveConfigValue(const char *name, std::int32_t explicitValue,
                                std::int32_t defaultValue) {
  const char *envValue = std::getenv(name);
  if (!envValue || !*envValue)
    return explicitValue > 0 ? explicitValue : defaultValue;

  char *end = nullptr;
  long parsed = std::strtol(envValue, &end, 10);
  if (!end || *end != '\0' || parsed <= 0)
    fail(name, "expected a positive integer environment value");
  return static_cast<std::int32_t>(parsed);
}

PyObject *buildIntArgs(std::int32_t a, std::int32_t b, std::int32_t c,
                       std::int32_t d) {
  PyObject *args = PyTuple_New(4);
  if (!args)
    fail("allocate Python arg tuple", fetchPythonError());

  const std::int32_t values[4] = {a, b, c, d};
  for (int i = 0; i < 4; ++i) {
    PyObject *value = PyLong_FromLong(values[i]);
    if (!value) {
      decref(args);
      fail("allocate Python integer arg", fetchPythonError());
    }
    if (PyTuple_SetItem(args, i, value) != 0) {
      decref(value);
      decref(args);
      fail("populate Python arg tuple", fetchPythonError());
    }
  }

  return args;
}

PyObject *buildSingleIntArgs(std::int32_t value) {
  PyObject *args = PyTuple_New(1);
  PyObject *valueArg = PyLong_FromLong(value);
  if (!args || !valueArg) {
    decref(args);
    decref(valueArg);
    fail("allocate Python integer arg", fetchPythonError());
  }
  if (PyTuple_SetItem(args, 0, valueArg) != 0) {
    decref(args);
    decref(valueArg);
    fail("populate Python arg tuple", fetchPythonError());
  }
  return args;
}

PyObject *callNoArgMethod(PyObject *receiver, const char *methodName) {
  PyObject *method = PyObject_GetAttrString(receiver, methodName);
  if (!method)
    fail(methodName, fetchPythonError());
  PyObject *args = PyTuple_New(0);
  if (!args) {
    decref(method);
    fail(methodName, fetchPythonError());
  }
  PyObject *result = PyObject_CallObject(method, args);
  decref(args);
  decref(method);
  if (!result)
    fail(methodName, fetchPythonError());
  return result;
}

PyObject *callIntMethodForResult(PyObject *receiver, const char *methodName,
                                 std::int32_t value) {
  PyObject *method = PyObject_GetAttrString(receiver, methodName);
  if (!method)
    fail(methodName, fetchPythonError());
  PyObject *args = buildSingleIntArgs(value);
  PyObject *result = PyObject_CallObject(method, args);
  decref(args);
  decref(method);
  if (!result)
    fail(methodName, fetchPythonError());
  return result;
}

void callIntMethod(PyObject *receiver, const char *methodName, std::int32_t value) {
  PyObject *result = callIntMethodForResult(receiver, methodName, value);
  decref(result);
}

void callPointerAndIntMethod(PyObject *receiver, const char *methodName,
                             const void *data, std::int32_t value) {
  PyObject *method = PyObject_GetAttrString(receiver, methodName);
  PyObject *args = PyTuple_New(2);
  PyObject *pointerArg = PyLong_FromVoidPtr(const_cast<void *>(data));
  PyObject *valueArg = PyLong_FromLong(value);
  if (!method || !args || !pointerArg || !valueArg) {
    decref(method);
    decref(args);
    decref(pointerArg);
    decref(valueArg);
    fail(methodName, fetchPythonError());
  }
  if (PyTuple_SetItem(args, 0, pointerArg) != 0) {
    decref(args);
    decref(method);
    decref(pointerArg);
    decref(valueArg);
    fail(methodName, fetchPythonError());
  }
  if (PyTuple_SetItem(args, 1, valueArg) != 0) {
    decref(args);
    decref(method);
    decref(valueArg);
    fail(methodName, fetchPythonError());
  }
  PyObject *result = PyObject_CallObject(method, args);
  decref(args);
  decref(method);
  if (!result)
    fail(methodName, fetchPythonError());
  decref(result);
}

void ensureInitializedLocked(std::int32_t numCores, std::int32_t arraysPerCore,
                             std::int32_t arrayRows, std::int32_t arrayCols) {
  BridgeState &state = getState();
  if (state.initialized) {
    ++state.clients;
    return;
  }

  if (!Py_IsInitialized())
    Py_Initialize();

  const char *simPath = std::getenv("ANALOG_GOLEM_SIM_PYTHON_DIR");
  if (!simPath || !*simPath)
    simPath = ANALOG_GOLEM_SIM_PYTHON_DIR;

  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *module = importModuleWithPath(simPath, "core_manager");
  PyObject *runtimeClass = PyObject_GetAttrString(module, "CoreManagerRuntime");
  decref(module);
  if (!runtimeClass)
    fail("load CoreManagerRuntime", fetchPythonError());

  PyObject *initializeMethod = PyObject_GetAttrString(runtimeClass, "initialize");
  if (!initializeMethod) {
    decref(runtimeClass);
    fail("load CoreManagerRuntime.initialize", fetchPythonError());
  }

  const std::int32_t resolvedNumCores =
      resolveConfigValue("NUM_CORES", numCores, 1);
  const std::int32_t resolvedArraysPerCore =
      resolveConfigValue("ARRAYS_PER_CORE", arraysPerCore, 1);
  const std::int32_t resolvedArrayRows =
      resolveConfigValue("ARRAY_ROWS", arrayRows, 1);
  const std::int32_t resolvedArrayCols =
      resolveConfigValue("ARRAY_COLS", arrayCols, 1);

  PyObject *args = buildIntArgs(resolvedNumCores, resolvedArraysPerCore,
                                resolvedArrayRows, resolvedArrayCols);
  PyObject *manager = PyObject_CallObject(initializeMethod, args);
  decref(args);
  decref(initializeMethod);
  if (!manager) {
    decref(runtimeClass);
    fail("initialize CoreManagerRuntime", fetchPythonError());
  }

  state.runtimeClass = runtimeClass;
  state.manager = manager;
  state.initialized = true;
  state.clients = 1;
  PyGILState_Release(gil);
}

} // namespace

void initializeBridge(std::int32_t numCores, std::int32_t arraysPerCore,
                      std::int32_t arrayRows, std::int32_t arrayCols) {
  std::lock_guard<std::mutex> lock(getMutex());
  ensureInitializedLocked(numCores, arraysPerCore, arrayRows, arrayCols);
}

void shutdownBridge() {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    return;
  if (state.clients > 1) {
    --state.clients;
    return;
  }

  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *result = callNoArgMethod(state.runtimeClass, "shutdown");
  decref(result);
  decref(state.manager);
  decref(state.runtimeClass);
  state.manager = nullptr;
  state.runtimeClass = nullptr;
  state.initialized = false;
  state.clients = 0;
  PyGILState_Release(gil);
}

void setActiveCore(std::int32_t coreIndex) {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    fail("setActiveCore", "bridge is not initialized");

  PyGILState_STATE gil = PyGILState_Ensure();
  callIntMethod(state.manager, "set_active_core", coreIndex);
  PyGILState_Release(gil);
}

void clearActiveCore() {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    return;

  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *result = callNoArgMethod(state.manager, "clear_active_core");
  decref(result);
  PyGILState_Release(gil);
}

void recordMvmSet(const float *data, std::int32_t rawArrayId) {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    fail("recordMvmSet", "bridge is not initialized");

  PyGILState_STATE gil = PyGILState_Ensure();
  callPointerAndIntMethod(state.manager, "record_mvm_set", data, rawArrayId);
  PyGILState_Release(gil);
}

void recordMvmLoad(const float *data, std::int32_t rawArrayId) {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    fail("recordMvmLoad", "bridge is not initialized");

  PyGILState_STATE gil = PyGILState_Ensure();
  callPointerAndIntMethod(state.manager, "record_mvm_load", data, rawArrayId);
  PyGILState_Release(gil);
}

void recordMvmCompute(std::int32_t rawArrayId) {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    fail("recordMvmCompute", "bridge is not initialized");

  PyGILState_STATE gil = PyGILState_Ensure();
  callIntMethod(state.manager, "record_mvm_compute", rawArrayId);
  PyGILState_Release(gil);
}

void copyMvmStore(float *dst, std::uint64_t elementCount,
                  std::int32_t rawArrayId) {
  std::lock_guard<std::mutex> lock(getMutex());
  BridgeState &state = getState();
  if (!state.initialized)
    fail("copyMvmStore", "bridge is not initialized");

  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject *result =
      callIntMethodForResult(state.manager, "get_output_array_from_core",
                             rawArrayId);

  Py_buffer buffer;
  std::memset(&buffer, 0, sizeof(buffer));
  if (PyObject_GetBuffer(result, &buffer, kPyBufferSimple) != 0) {
    decref(result);
    fail("read simulator output buffer", fetchPythonError());
  }

  const std::size_t expectedBytes = elementCount * sizeof(float);
  if (buffer.len != static_cast<Py_ssize_t>(expectedBytes)) {
    PyBuffer_Release(&buffer);
    decref(result);
    fail("copy simulator output",
         "unexpected output size returned by Python simulator");
  }

  std::memcpy(dst, buffer.buf, expectedBytes);
  PyBuffer_Release(&buffer);
  decref(result);
  PyGILState_Release(gil);
}

} // namespace analog::golem_sim
