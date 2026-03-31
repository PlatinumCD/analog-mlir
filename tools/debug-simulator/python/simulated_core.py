import ctypes
import queue
import threading
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


class TaskKind(Enum):
    WEIGHT = 0
    LAYER = 1


class SimulatedCore:
    """Represents a single simulated analog core backed by a Python worker thread."""

    def __init__(
        self,
        core_id: int,
        completion_callback: Callable[[], None],
        num_arrays: int = 1,
        array_shape: Sequence[int] = (32, 32),
    ):
        self.core_id = core_id
        self._queue: queue.Queue[Optional[Tuple[int, Callable[[int], None], TaskKind, int]]] = queue.Queue()
        self._completion_callback = completion_callback
        self._run_weight: Optional[Callable[[int], None]] = None
        self._run_layer: Optional[Callable[[int], None]] = None
        self._shutdown = False
        self.array_shape: Tuple[int, ...] = tuple(array_shape)
        self.row_length = int(self.array_shape[0]) if len(self.array_shape) > 0 else 0
        self.arrays = [
            np.zeros(self.array_shape, dtype=np.float32) for _ in range(num_arrays)
        ]
        self.input_vectors = [
            np.zeros((self.row_length,), dtype=np.float32) for _ in range(num_arrays)
        ]
        self.output_vectors = [
            np.zeros((self.row_length,), dtype=np.float32) for _ in range(num_arrays)
        ]
        self._array_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(),
            daemon=True,
            name=f"analog-sim-core-{core_id}",
        )

    def start(self) -> None:
        """Launch the worker thread if it has not already started."""

        if not self._thread.is_alive():
            self._thread.start()

    def set_run_weight(self, callback: Callable[[int], None]) -> None:
        """Registers the callable that executes analog weight dispatches."""

        self._run_weight = callback

    def set_run_layer(self, callback: Callable[[int], None]) -> None:
        self._run_layer = callback

    def dispatch(self, weight_id: int) -> None:
        """Enqueue a weight dispatch request for this core."""

        if self._run_weight:
            self._queue.put((weight_id, self._run_weight, TaskKind.WEIGHT, weight_id))

    def dispatch_layer(self, layer_id: int) -> None:
        if self._run_layer:
            self._queue.put((self.core_id, self._run_layer, TaskKind.LAYER, layer_id))

    def shutdown(self) -> None:
        """Gracefully stops the worker thread."""

        if self._shutdown:
            return
        self._queue.put(None)
        self._thread.join(timeout=0.1)
        self._shutdown = True

    def load_array_data(self, raw_array_id: int, data_ptr: int) -> None:
        if raw_array_id < 0 or raw_array_id >= len(self.arrays):
            return
        if not data_ptr:
            return

        total_elems = int(np.prod(self.array_shape))
        buffer_type = ctypes.c_float * total_elems
        try:
            raw_buffer = ctypes.cast(data_ptr, ctypes.POINTER(buffer_type)).contents
        except (ValueError, TypeError):
            return
        numpy_view = np.ctypeslib.as_array(raw_buffer).reshape(self.array_shape)
        with self._array_lock:
            np.copyto(self.arrays[raw_array_id], numpy_view)
        print(f"[python core {self.core_id}] loaded array {raw_array_id}:")
        print(self.arrays[raw_array_id])
        print()

    def load_input_vector(self, raw_array_id: int, data_ptr: int) -> None:
        if raw_array_id < 0 or raw_array_id >= len(self.input_vectors):
            return
        if not data_ptr or self.row_length == 0:
            return

        buffer_type = ctypes.c_float * self.row_length
        try:
            raw_buffer = ctypes.cast(data_ptr, ctypes.POINTER(buffer_type)).contents
        except (ValueError, TypeError):
            return
        numpy_view = np.ctypeslib.as_array(raw_buffer)
        with self._array_lock:
            np.copyto(self.input_vectors[raw_array_id], numpy_view)
        print(f"[python core {self.core_id}] loaded input vector {raw_array_id}:")
        print(self.input_vectors[raw_array_id])
        print()

    def store_output_vector(self, raw_array_id: int, data_ptr: int) -> None:
        if raw_array_id < 0 or raw_array_id >= len(self.output_vectors):
            return
        if not data_ptr or self.row_length == 0:
            return

        buffer_type = ctypes.c_float * self.row_length
        try:
            raw_buffer = ctypes.cast(data_ptr, ctypes.POINTER(buffer_type)).contents
        except (ValueError, TypeError):
            return
        with self._array_lock:
            np.copyto(np.ctypeslib.as_array(raw_buffer), self.output_vectors[raw_array_id])
        print(f"[python core {self.core_id}] stored output vector {raw_array_id}:")
        print(self.output_vectors[raw_array_id])
        print()

    def _run_loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return

            job_id, callback, kind, payload = task
            if job_id == self.core_id and callback:
                callback(payload)
            self._completion_callback(kind)
            self._queue.task_done()
