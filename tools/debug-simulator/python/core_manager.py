import ctypes
import os
import threading

import numpy as np

from simulated_core import SimulatedCore


def _read_env_int(name, default):
    return int(os.environ.get(name, str(default)))


class CoreManager:
    def __init__(self, num_cores, arrays_per_core, array_rows, array_cols):
        self.num_cores = num_cores
        self.arrays_per_core = arrays_per_core
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.active_core = None
        print(
            "[core manager] initializing "
            f"{num_cores} cores with {arrays_per_core} arrays/core "
            f"({array_rows}x{array_cols})"
        )
        self.cores = []
        for core_index in range(num_cores):
            core = SimulatedCore(
                arrays_per_core,
                array_rows,
                array_cols,
                core_index=core_index,
            )
            self.cores.append(core)
            print(f"[core manager] initialized core {core_index}")
        self._shutdown_event = threading.Event()
        self.threads = []

        for core_index in range(num_cores):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(core_index,),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def _worker_loop(self, core_index):
        (void := core_index)
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait()

    def set_active_core(self, core_index):
        if not 0 <= core_index < self.num_cores:
            raise IndexError(
                f"core_index {core_index} out of bounds for {self.num_cores} cores"
            )
        self.active_core = core_index
        print(f"[core manager] active core set to {core_index}")

    def clear_active_core(self):
        self.active_core = None
        print("[core manager] active core cleared")

    def record_mvm_set(self, data_ptr, raw_array_id):
        if self.active_core is None:
            raise ValueError("active core is not set")
        if not 0 <= raw_array_id < self.arrays_per_core:
            raise IndexError(
                f"raw_array_id {raw_array_id} out of bounds for "
                f"{self.arrays_per_core} arrays per core"
            )

        buffer_type = ctypes.c_float * (self.array_rows * self.array_cols)
        raw_buffer = ctypes.cast(data_ptr, ctypes.POINTER(buffer_type)).contents
        data = np.ctypeslib.as_array(raw_buffer).reshape(
            self.array_rows, self.array_cols
        )
        self.cores[self.active_core].set_array(data, raw_array_id)

    def record_mvm_load(self, data_ptr, raw_array_id):
        if self.active_core is None:
            raise ValueError("active core is not set")
        if not 0 <= raw_array_id < self.arrays_per_core:
            raise IndexError(
                f"raw_array_id {raw_array_id} out of bounds for "
                f"{self.arrays_per_core} arrays per core"
            )

        buffer_type = ctypes.c_float * self.array_rows
        raw_buffer = ctypes.cast(data_ptr, ctypes.POINTER(buffer_type)).contents
        data = np.ctypeslib.as_array(raw_buffer)
        self.cores[self.active_core].load_input(data, raw_array_id)

    def record_mvm_compute(self, raw_array_id):
        if self.active_core is None:
            raise ValueError("active core is not set")
        if not 0 <= raw_array_id < self.arrays_per_core:
            raise IndexError(
                f"raw_array_id {raw_array_id} out of bounds for "
                f"{self.arrays_per_core} arrays per core"
            )

        self.cores[self.active_core].compute(raw_array_id)

    def shutdown(self):
        self._shutdown_event.set()
        for thread in self.threads:
            thread.join()


class CoreManagerRuntime:
    _core_manager = None

    @classmethod
    def initialize(
        cls,
        num_cores=None,
        arrays_per_core=None,
        array_rows=None,
        array_cols=None,
    ):
        if cls._core_manager is None:
            resolved_num_cores = (
                num_cores if num_cores is not None else _read_env_int("NUM_CORES", 1)
            )
            resolved_arrays_per_core = (
                arrays_per_core
                if arrays_per_core is not None
                else _read_env_int("ARRAYS_PER_CORE", 1)
            )
            resolved_array_rows = (
                array_rows
                if array_rows is not None
                else _read_env_int("ARRAY_ROWS", 1)
            )
            resolved_array_cols = (
                array_cols
                if array_cols is not None
                else _read_env_int("ARRAY_COLS", 1)
            )
            cls._core_manager = CoreManager(
                num_cores=resolved_num_cores,
                arrays_per_core=resolved_arrays_per_core,
                array_rows=resolved_array_rows,
                array_cols=resolved_array_cols,
            )
            print("[core manager runtime] initialized")
        return cls._core_manager

    @classmethod
    def get_core_manager(cls):
        return cls.initialize()

    @classmethod
    def shutdown(cls):
        if cls._core_manager is not None:
            cls._core_manager.shutdown()
            cls._core_manager = None
