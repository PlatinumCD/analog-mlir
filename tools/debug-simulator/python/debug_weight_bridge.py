import atexit
import ctypes
import os
import threading

from simulated_core import SimulatedCore, TaskKind

_initialized = False
_worker_count = 0
_loaded_so_path = None
_loaded_library = None
_run_weight = None
_worker_cores: list[SimulatedCore] = []
_inflight_weights = 0
_inflight_layers = 0
_drain_condition = threading.Condition()
_shutdown = False
ARRAYS_PER_CORE = max(1, int(os.environ.get("ARRAYS_PER_CORE", "1")))
array_rows = int(os.environ.get("ARRAY_ROWS", "32"))
array_cols = int(os.environ.get("ARRAY_COLS", "32"))
ARRAY_SHAPE = (array_rows, array_cols)


def _completion_callback(kind: TaskKind) -> None:
    global _inflight_weights, _inflight_layers
    with _drain_condition:
        if kind == TaskKind.WEIGHT:
            _inflight_weights -= 1
        else:
            _inflight_layers -= 1
        if _inflight_weights == 0 and _inflight_layers == 0:
            _drain_condition.notify_all()


def _shutdown_workers():
    global _shutdown
    if _shutdown:
        return
    _shutdown = True
    for core in _worker_cores:
        core.shutdown()


def initialize_bridge(worker_count: int):
    global _initialized, _worker_count, _worker_cores
    if _initialized:
        print("[python bridge] initialize_bridge: already initialized")
        return True

    print(f"[python bridge] initialize_bridge: creating {worker_count} workers")
    _worker_count = worker_count
    _worker_cores = [
        SimulatedCore(
            worker_id,
            _completion_callback,
            num_arrays=ARRAYS_PER_CORE,
            array_shape=ARRAY_SHAPE,
        )
        for worker_id in range(worker_count)
    ]
    for core in _worker_cores:
        core.start()

    atexit.register(_shutdown_workers)
    _initialized = True
    return True


def bind_shared_object(so_path: str):
    global _loaded_so_path, _loaded_library, _run_weight

    if _loaded_so_path == so_path and _loaded_library is not None:
        return True

    mode = os.RTLD_LOCAL
    if hasattr(os, "RTLD_LAZY"):
        mode |= os.RTLD_LAZY

    _loaded_library = ctypes.CDLL(so_path, mode=mode)
    _run_weight = _loaded_library.analog_run_weight
    _run_weight.argtypes = [ctypes.c_int32]
    _run_weight.restype = None

    _loaded_so_path = so_path
    for core in _worker_cores:
        core.set_run_weight(_run_weight)
        core.set_run_layer(_layer_callback)
    return True


def dispatch_weight(weight_id: int):
    global _inflight_weights
    with _drain_condition:
        _inflight_weights += len(_worker_cores)
    for core in _worker_cores:
        core.dispatch(weight_id)
    return True


def wait_weights():
    with _drain_condition:
        while _inflight_weights > 0:
            _drain_condition.wait()
    print("[python bridge] wait_weights: all queued weights drained")
    return True


def _layer_callback(layer_id: int) -> None:
    print(f"[python bridge] running layer {layer_id}")


def dispatch_layer(layer_id: int) -> bool:
    global _inflight_layers
    with _drain_condition:
        _inflight_layers += len(_worker_cores)
    for core in _worker_cores:
        core.dispatch_layer(layer_id)
    return True


def wait_layers() -> bool:
    with _drain_condition:
        while _inflight_layers > 0:
            _drain_condition.wait()
    print("[python bridge] wait_layers: all queued layers drained")
    return True


def record_mvm_set(data_ptr: int, raw_array_id: int):
    if not isinstance(raw_array_id, int):
        return False
    for core in _worker_cores:
        core.load_array_data(raw_array_id, data_ptr)
    return True


def record_mvm_load(data_ptr: int, raw_array_id: int):
    if not isinstance(raw_array_id, int):
        return False
    for core in _worker_cores:
        core.load_input_vector(raw_array_id, data_ptr)
    return True


def record_mvm_compute(raw_array_id: int):
    if not isinstance(raw_array_id, int):
        return False
    print(f"[python bridge] mvm_compute array {raw_array_id} invoked")
    return True


def record_mvm_store(data_ptr: int, raw_array_id: int):
    if not isinstance(raw_array_id, int):
        return False
    for core in _worker_cores:
        core.store_output_vector(raw_array_id, data_ptr)
    return True
