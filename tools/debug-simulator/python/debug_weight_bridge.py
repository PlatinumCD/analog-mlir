import atexit
import ctypes
import os
import queue
import threading

_initialized = False
_worker_count = 0
_loaded_so_path = None
_loaded_library = None
_run_weight = None
_worker_queues = []
_worker_threads = []
_inflight = 0
_drain_condition = threading.Condition()
_shutdown = False


def _worker_loop(worker_id: int, task_queue: "queue.Queue[int]"):
    global _inflight
    print(f"[python bridge] worker[{worker_id}] started")
    while True:
        weight_id = task_queue.get()
        if weight_id is None:
            print(f"[python bridge] worker[{worker_id}] shutting down")
            task_queue.task_done()
            return

        print(
            f"[python bridge] worker[{worker_id}] handling weight {weight_id} "
            f"from {_loaded_so_path}"
        )
        print(
            f"[python bridge] worker[{worker_id}] would call "
            f"analog_run_weight({weight_id})"
        )
        # _run_weight(weight_id)

        with _drain_condition:
            _inflight -= 1
            if _inflight == 0:
                _drain_condition.notify_all()
        task_queue.task_done()


def _shutdown_workers():
    global _shutdown
    if _shutdown:
        return
    _shutdown = True
    for worker_queue in _worker_queues:
        worker_queue.put(None)
    for worker in _worker_threads:
        worker.join(timeout=0.1)


def initialize_bridge(worker_count: int):
    global _initialized, _worker_count, _worker_queues, _worker_threads
    if _initialized:
        print("[python bridge] initialize_bridge: already initialized")
        return True

    print(f"[python bridge] initialize_bridge: creating {worker_count} workers")
    _worker_count = worker_count
    _worker_queues = [queue.Queue() for _ in range(worker_count)]
    _worker_threads = [
        threading.Thread(
            target=_worker_loop,
            args=(worker_id, worker_queue),
            daemon=True,
            name=f"analog-sim-core-{worker_id}",
        )
        for worker_id, worker_queue in enumerate(_worker_queues)
    ]
    for worker in _worker_threads:
        worker.start()

    atexit.register(_shutdown_workers)
    _initialized = True
    return True


def bind_shared_object(so_path: str):
    global _loaded_so_path, _loaded_library, _run_weight

    print(f"[python bridge] bind_shared_object: {so_path}")
    if _loaded_so_path == so_path and _loaded_library is not None:
        print("[python bridge] bind_shared_object: already bound")
        return True

    mode = os.RTLD_LOCAL
    if hasattr(os, "RTLD_LAZY"):
        mode |= os.RTLD_LAZY

    _loaded_library = ctypes.CDLL(so_path, mode=mode)
    _run_weight = _loaded_library.analog_run_weight
    _run_weight.argtypes = [ctypes.c_int32]
    _run_weight.restype = None

    _loaded_so_path = so_path
    print("[python bridge] bind_shared_object: resolved analog_run_weight")
    return True


def dispatch_weight(weight_id: int):
    global _inflight
    worker_id = weight_id % _worker_count
    with _drain_condition:
        _inflight += 1
    print(
        f"[python bridge] dispatch_weight: enqueue weight {weight_id} "
        f"to worker[{worker_id}]"
    )
    _worker_queues[worker_id].put(weight_id)
    return True


def wait_weights():
    with _drain_condition:
        while _inflight > 0:
            _drain_condition.wait()
    print("[python bridge] wait_weights: all queued weights drained")
    return True
